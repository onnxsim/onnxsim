#!/usr/bin/env bash
# Build-test the Rust crates the way a downstream consumer sees them: package
# them with `cargo package`, unpack the archives *outside* this repository, and
# build a throwaway crate against the unpacked copies.
#
# Why this exists: `cargo publish --no-verify` (see "Publishing" in
# rust/README.md) means nothing in CI ever compiles what actually ships. The
# packaged crate is just the crate directory -- no C++ sources, no submodules,
# no monorepo around it -- so a mistake that only shows up there (a file missing
# from the package, a path dependency that does not resolve once rewritten to a
# registry dependency, a build script that assumes ../.. is the onnxsim source
# tree) is invisible to the in-tree `cargo build` the `build` job runs.
#
# Usage:
#   scripts/test_rust_package_standalone.sh [options]
#
# Options:
#   --lib-dir DIR[:DIR...]  Link a pre-built onnxsim_c from these directories
#                           (ONNXSIM_LIB_DIR mode) and *run* the consumer, so
#                           the packaged crates are linked and executed, not
#                           just type-checked. "auto" searches this repo's
#                           rust/target for a library left by a previous
#                           `cargo build`.
#   --source-dir DIR        Build the native library from source with CMake
#                           against the onnxsim checkout at DIR
#                           (ONNXSIM_SOURCE_DIR mode) and run the consumer.
#                           Slow; honors ONNXSIM_PREBUILT_ORT=1.
#   --model PATH            Also simplify this ONNX model from the consumer
#                           binary (only used when the native library is
#                           linked).
#   --workdir DIR           Unpack into DIR instead of a fresh mktemp -d.
#                           Must be outside this repository.
#   --keep                  Do not delete the work directory on exit.
#
# With neither --lib-dir nor --source-dir the script runs in check-only mode:
# it type-checks the unpacked crates with ONNXSIM_NO_BUILD=1 (no native build,
# ~a minute) and asserts that a build with no mode selected fails fast with an
# actionable message instead of trying to compile the C++ stack from a
# directory that does not contain it.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)
RUST_DIR="$ROOT_DIR/rust"

LIB_DIR=""
SOURCE_DIR=""
MODEL=""
WORKDIR=""
KEEP=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --lib-dir) LIB_DIR="${2:?--lib-dir needs a value}"; shift 2 ;;
    --source-dir) SOURCE_DIR="${2:?--source-dir needs a value}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
    --workdir) WORKDIR="${2:?--workdir needs a value}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The consumer binary runs from the work directory, so a relative model path
# would resolve against the wrong directory once it gets there.
if [ -n "$MODEL" ]; then
  [ -f "$MODEL" ] || { echo "error: model $MODEL does not exist" >&2; exit 2; }
  MODEL=$(cd -- "$(dirname -- "$MODEL")" &> /dev/null && pwd)/$(basename -- "$MODEL")
fi

if [ -n "$LIB_DIR" ] && [ -n "$SOURCE_DIR" ]; then
  echo "error: --lib-dir and --source-dir are mutually exclusive" >&2
  exit 2
fi

# `auto`: reuse whatever a previous in-tree `cargo build` already produced, so
# CI can link against the native library it just spent an hour compiling
# instead of building it a second time.
if [ "$LIB_DIR" = "auto" ]; then
  # cargo leaves one build directory per build-script run, and only the one
  # that actually built the library has it, so take the first that does.
  build_root=""
  while IFS= read -r candidate; do
    if [ -n "$(find "$candidate" \( -name 'libonnxsim_c.*' -o -name 'onnxsim_c.dll' \) -print -quit 2>/dev/null)" ]; then
      build_root="$candidate"
      break
    fi
  # Match both cargo layouts: older releases use build/onnxsim-sys-<hash>/out,
  # newer ones build/onnxsim-sys/<hash>/out.
  done < <(find "$RUST_DIR/target" -type d -path '*/onnxsim-sys*/out/build' 2>/dev/null)

  if [ -z "$build_root" ]; then
    echo "error: --lib-dir auto found no built onnxsim_c under $RUST_DIR/target;" >&2
    echo "       run a native \`cargo build\` in rust/ first, or pass an explicit --lib-dir" >&2
    exit 1
  fi

  # onnxsim_c links against ONNX Runtime & friends, which sit in sibling
  # directories of the same build tree; list every directory holding a shared
  # library so both the linker and the loader can resolve them too.
  LIB_DIR=$(find "$build_root" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.dll' \) 2>/dev/null |
            xargs -r -n1 dirname | sort -u | paste -sd: -)
  echo "--lib-dir auto resolved to $LIB_DIR"
fi

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$RUST_DIR/Cargo.toml" | head -n 1)
if [ -z "$VERSION" ]; then
  echo "error: could not read the workspace version from $RUST_DIR/Cargo.toml" >&2
  exit 1
fi

if [ -n "$WORKDIR" ]; then
  mkdir -p "$WORKDIR"
  WORKDIR=$(cd -- "$WORKDIR" &> /dev/null && pwd)
else
  WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/onnxsim-rust-standalone.XXXXXX")
fi

# The whole point is to build with no onnxsim source tree around; a work
# directory inside the repository would silently defeat that (the build script
# walks up from the crate directory to find the C++ sources).
case "$WORKDIR/" in
  "$ROOT_DIR"/*) echo "error: work directory $WORKDIR is inside $ROOT_DIR" >&2; exit 2 ;;
esac

cleanup() {
  if [ "$KEEP" -eq 0 ]; then
    rm -rf "$WORKDIR"
  else
    echo "kept work directory: $WORKDIR"
  fi
}
trap cleanup EXIT

echo "=== packaging onnxsim-sys $VERSION and onnxsim $VERSION ==="
# Drop any archive left by an earlier run so a stale (or half-written) .crate
# can never be what gets tested.
rm -f "$RUST_DIR/target/package/onnxsim-sys-$VERSION.crate" \
      "$RUST_DIR/target/package/onnxsim-$VERSION.crate"
# --no-verify for the same reason `cargo publish` needs it: the verification
# build happens in a sandbox with no C++ sources. That build is what this
# script replaces, in a way that actually works. ONNXSIM_NO_BUILD keeps any
# incidental build-script run from starting a native build.
(
  cd "$RUST_DIR"
  ONNXSIM_NO_BUILD=1 cargo package --no-verify --allow-dirty -p onnxsim-sys -p onnxsim
)

for crate in "onnxsim-sys-$VERSION" "onnxsim-$VERSION"; do
  archive="$RUST_DIR/target/package/$crate.crate"
  [ -f "$archive" ] || { echo "error: $archive was not produced" >&2; exit 1; }
  tar xzf "$archive" -C "$WORKDIR"
  [ -d "$WORKDIR/$crate" ] || { echo "error: $archive did not unpack to $crate" >&2; exit 1; }
done

# Sanity check the premise: nothing that looks like an onnxsim checkout is
# visible from the unpacked crates, so the build script cannot fall back to it.
for marker in "$WORKDIR/CMakeLists.txt" "$WORKDIR/onnxsim"; do
  if [ -e "$marker" ]; then
    echo "error: $marker exists; $WORKDIR is not a source-tree-free environment" >&2
    exit 1
  fi
done

# `cargo package` rewrites onnxsim's path dependency on onnxsim-sys into a
# registry dependency, so a plain build of the unpacked crate would pull
# onnxsim-sys from crates.io -- the published copy, not the one being tested.
# Patch it back to the freshly unpacked sibling. Putting the patch in
# .cargo/config.toml (rather than a manifest) applies it to every cargo
# invocation below, including ones run inside the unpacked crates themselves.
mkdir -p "$WORKDIR/.cargo"
cat > "$WORKDIR/.cargo/config.toml" <<EOF
[patch.crates-io]
onnxsim-sys = { path = "$WORKDIR/onnxsim-sys-$VERSION" }
EOF

# A downstream crate that depends on onnxsim exactly as the README documents.
mkdir -p "$WORKDIR/consumer/src"
cat > "$WORKDIR/consumer/Cargo.toml" <<EOF
[package]
name = "onnxsim-standalone-consumer"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
onnxsim = { path = "$WORKDIR/onnxsim-$VERSION", version = "$VERSION" }

# Not a member of any surrounding workspace.
[workspace]
EOF

cat > "$WORKDIR/consumer/src/main.rs" <<'EOF'
//! Smoke test for the packaged crates, from the outside.
//!
//! Type-checks against the public API in check-only mode; when the native
//! library is linked it also calls into it (and simplifies a model if one is
//! passed as an argument).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let optimizers = onnxsim::list_optimizers();
    println!("list_optimizers() -> {} passes", optimizers.len());
    assert!(!optimizers.is_empty(), "expected a non-empty optimizer list");

    // Exercise a little of the safe API that does not need a model.
    let _opts = onnxsim::Options::new()
        .shape_inference(true)
        .skip_optimizer(optimizers[0].clone())
        .tensor_size_threshold(1024 * 1024);

    if let Some(path) = std::env::args().nth(1) {
        let model = std::fs::read(&path)?;
        let simplified = onnxsim::simplify(&model)?;
        assert!(!simplified.is_empty(), "simplify() returned no model");
        print!("{}", onnxsim::model_info_diff(&model, &simplified)?);
        println!("simplified {path}: {} -> {} bytes", model.len(), simplified.len());
    }
    Ok(())
}
EOF

if [ -n "$LIB_DIR" ] || [ -n "$SOURCE_DIR" ]; then
  if [ -n "$LIB_DIR" ]; then
    echo "=== native mode: linking pre-built onnxsim_c from $LIB_DIR ==="
    export ONNXSIM_LIB_DIR="$LIB_DIR"
    # rpath is emitted by the build script, but keep the loader happy on
    # platforms/configurations where it is not honored.
    export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_FALLBACK_LIBRARY_PATH="$LIB_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  else
    SOURCE_DIR=$(cd -- "$SOURCE_DIR" &> /dev/null && pwd)
    echo "=== native mode: building onnxsim_c from source at $SOURCE_DIR ==="
    export ONNXSIM_SOURCE_DIR="$SOURCE_DIR"
  fi

  echo "--- running the consumer binary ---"
  (
    cd "$WORKDIR/consumer"
    if [ -n "$MODEL" ]; then
      cargo run --release -- "$MODEL"
    else
      cargo run --release
    fi
  )

  echo "--- running the packaged crate's own tests ---"
  (
    cd "$WORKDIR/onnxsim-$VERSION"
    cargo test --release
    cargo test --release -- --ignored list_optimizers_is_non_empty
  )
else
  echo "=== check-only mode (ONNXSIM_NO_BUILD=1, no native build) ==="
  echo "--- type-checking the unpacked crates (lib, examples, tests) ---"
  (
    cd "$WORKDIR/onnxsim-$VERSION"
    ONNXSIM_NO_BUILD=1 cargo check --all-targets
  )
  echo "--- type-checking a downstream consumer of the unpacked crate ---"
  (
    cd "$WORKDIR/consumer"
    ONNXSIM_NO_BUILD=1 cargo check
  )

  # With no mode selected there is no way to produce the native library out
  # here, so the build script must say so immediately. Before it did this, it
  # downloaded the ONNX Runtime source into whatever directory happened to sit
  # above the crate and only failed once CMake was handed a path with no
  # CMakeLists.txt.
  echo "--- asserting an unconfigured build fails fast with an actionable error ---"
  log="$WORKDIR/unconfigured-build.log"
  if (cd "$WORKDIR/consumer" && cargo check --offline) > "$log" 2>&1; then
    echo "error: a build with no ONNXSIM_* mode selected unexpectedly succeeded" >&2
    exit 1
  fi
  for expected in ONNXSIM_LIB_DIR ONNXSIM_SOURCE_DIR ONNXSIM_NO_BUILD; do
    grep -q "$expected" "$log" || {
      echo "error: the failure message does not mention $expected:" >&2
      cat "$log" >&2
      exit 1
    }
  done
  # ...and it must not have started fetching ONNX Runtime into the work dir.
  if [ -e "$WORKDIR/third_party" ]; then
    echo "error: the failed build created $WORKDIR/third_party" >&2
    exit 1
  fi
fi

echo "=== standalone package build test passed (onnxsim $VERSION) ==="
