# onnxsim — Rust bindings

Safe Rust bindings to the [ONNX Simplifier](https://github.com/onnxsim/onnxsim).
Simplify ONNX models (shape inference + constant folding) directly from Rust,
using the same C++ core as the Python package and the CLI — no need to shell out
to `onnxsim` or embed a Python interpreter.

This addresses [onnxsim/onnxsim#292](https://github.com/onnxsim/onnxsim/issues/292),
which requested a Rust wrapper so importers such as [Burn](https://github.com/tracel-ai/burn),
[tract](https://github.com/sonos/tract) and [wonnx](https://github.com/webonnx/wonnx)
can simplify models as part of their own pipelines.

## Layout

| Crate         | Role                                                             |
| ------------- | --------------------------------------------------------------- |
| `onnxsim`     | Safe, idiomatic API. Depend on this.                            |
| `onnxsim-sys` | Raw FFI declarations + the build script that links the C core.  |

Both wrap `onnxsim/capi/onnxsim_c_api.h`, a small C ABI over the C++ simplifier.

## Usage

```toml
[dependencies]
onnxsim = { git = "https://github.com/onnxsim/onnxsim", subdir = "rust/onnxsim" }
```

In-memory (serialized `ModelProto` bytes, e.g. from the `prost`/`protobuf`
generated ONNX types, or straight from disk):

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    std::fs::write("model.opt.onnx", &simplified)?;
    Ok(())
}
```

File in, file out:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    onnxsim::simplify_path("model.onnx", "model.opt.onnx")?;
    Ok(())
}
```

With options:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let opts = onnxsim::Options::new()
        .shape_inference(false)                       // skip if it crashes on your model
        .skip_optimizer("eliminate_nop_transpose")    // keep a specific pass off
        .extra_optimizer("defuse_matmul_integer_to_float") // opt into a non-default pass
        .tensor_size_threshold(512 * 1024 * 1024);
    let simplified = onnxsim::simplify_with(&model, &opts)?;
    Ok(())
}
```

List the optimizer passes you can skip, and the ones you can opt into with
[`Options::extra_optimizer`] (off by default -- typically a graph-shape rewrite
rather than a pure node reduction or fusion):

```rust
for name in onnxsim::list_optimizers() {
    println!("{name}");
}
for name in onnxsim::list_other_optimizers() {
    println!("{name}");
}
```

Print the before/after difference, the same op-count and model-size summary the
Python CLI shows after simplifying:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    print!("{}", onnxsim::model_info_diff(&model, &simplified)?);
    Ok(())
}
```

For the specific nodes and values that changed rather than just the aggregate
counts, use `graph_diff` instead: which nodes/values were removed, added, or
changed (matched by output tensor name), e.g. a Conv whose bias input got
folded into its weight.

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify(&model)?;
    print!("{}", onnxsim::graph_diff(&model, &simplified)?);
    Ok(())
}
```

Export a model to a standalone safetensors or GGUF archive (graph + weights in
one ecosystem-standard file) and import it back:

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    onnxsim::export_safetensors(&model, "model.onnx.safetensors")?;
    onnxsim::export_gguf(&model, "model.onnx.gguf")?;

    let model = onnxsim::import_safetensors("model.onnx.safetensors")?;
    let model = onnxsim::import_gguf("model.onnx.gguf")?;
    let _ = model;
    Ok(())
}
```

## Custom rewriter

Run your own graph-rewriting logic inside the simplification fixed point — the
Rust equivalent of the Python `custom_rewriter` parameter. The closure is called
each round with the current model as serialized `ModelProto` bytes and returns
`Ok(None)` (nothing changed this round), `Ok(Some(bytes))` (the rewritten
model), or `Err(..)` to abort. Because it is interleaved with the built-in
optimizer, shape inference and constant folding, a rewrite can unlock further
simplification and vice versa.

```rust
fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let model = std::fs::read("model.onnx")?;
    let simplified = onnxsim::simplify_with_rewriter(
        &model,
        &onnxsim::Options::new(),
        |bytes: &[u8]| {
            // Decode `bytes`, rewrite the graph, and return the new bytes,
            // or `Ok(None)` to report that nothing changed this round.
            let _ = bytes;
            Ok::<_, onnxsim::Error>(None)
        },
    )?;
    std::fs::write("model.opt.onnx", &simplified)?;
    Ok(())
}
```

## Building the native library

`onnxsim-sys` needs the `onnxsim_c` shared library. Its build script supports
three modes:

1. **From source (default).** Runs CMake to build the full onnxsim stack
   (ONNX Runtime, onnx-optimizer, protobuf). This is heavy the first time. Check
   out the git submodules first (for onnx-optimizer); the ONNX Runtime source is
   not a submodule and is downloaded automatically on the first build:

   ```sh
   git submodule update --init --recursive
   cargo build
   ```

   Set `ONNXSIM_SKIP_ORT_DOWNLOAD=1` to forbid the automatic download (the build
   then requires the ONNX Runtime source to already be present at
   `third_party/onnxruntime-1.29.0`).

   **Fast path — prebuilt ONNX Runtime.** To skip compiling ONNX Runtime from
   source, set `ONNXSIM_PREBUILT_ORT=1`. The build then links an official
   [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases)
   (downloaded and cached automatically) instead. onnx-optimizer, onnx and
   protobuf are still built from source, but the slowest dependency is skipped:

   ```sh
   ONNXSIM_PREBUILT_ORT=1 cargo build
   # optionally pin a version or reuse an already-extracted release:
   ONNXSIM_PREBUILT_ORT=1 ONNXSIM_ORT_VERSION=1.29.0 cargo build
   ONNXSIM_PREBUILT_ORT=1 ONNXSIM_ORT_HOME=/path/to/onnxruntime-linux-x64-1.29.0 cargo build
   ```

2. **Pre-built library.** If you already have `onnxsim_c` (and its dependencies)
   built, point the build script at the directory (or directories, `:`-separated)
   holding the shared libraries:

   ```sh
   ONNXSIM_LIB_DIR=/path/to/libs cargo build
   ```

   To produce it from this repo:

   ```sh
   cmake -B build -DONNXSIM_C_API=ON -DONNXSIM_BUILTIN_ORT=ON
   cmake --build build --target onnxsim_c
   ```

   Add `-DONNXSIM_PREBUILT_ORT=ON` to link an official ONNX Runtime release
   (downloaded and cached under the build tree) instead of compiling it from
   source. `-DONNXSIM_ORT_VERSION=<ver>` pins the release and
   `-DONNXSIM_ORT_HOME=<dir>` reuses an already-extracted one.

3. **Skip building** (for `cargo check` / docs.rs). Set `ONNXSIM_NO_BUILD=1`
   (docs.rs sets `DOCS_RS` automatically). The crate type-checks but cannot be
   linked into a runnable binary.

Mode 1 is the default only *inside* this repository, where the C++ sources sit
two directories above the crate. The published crate ships without them, so a
build from crates.io stops immediately with a message pointing at the three
modes rather than attempting a source build that cannot work; pick mode 2 or 3
there. [The standalone package build test](#standalone-package-build-test)
covers exactly that situation.

### Environment variables

| Variable             | Effect                                                        |
| -------------------- | ------------------------------------------------------------- |
| `ONNXSIM_NO_BUILD`        | Skip the native build entirely (type-check only).        |
| `ONNXSIM_LIB_DIR`         | `:`-separated dirs holding a pre-built `onnxsim_c`.      |
| `ONNXSIM_SOURCE_DIR`      | Override the onnxsim C++ source path (default `../..`).  |
| `ONNXSIM_SKIP_ORT_DOWNLOAD` | Forbid the automatic ONNX Runtime source download.    |
| `ONNXSIM_PREBUILT_ORT`    | Link a prebuilt ONNX Runtime release instead of building it. |
| `ONNXSIM_ORT_VERSION`     | Prebuilt release version to fetch (default `1.29.0`).    |
| `ONNXSIM_ORT_HOME`        | Use an already-extracted prebuilt release (no download). |
| `ONNXSIM_ORT_URL`         | Override the prebuilt release download URL.              |

## Examples & tests

```sh
cargo run --example simplify -- input.onnx output.onnx
cargo test          # builds/links the native lib, then runs the tests
```

Most of the unit tests exercise pure-Rust logic (the options builder, the
rewriter/executor trampolines, the DLPack conversions) and never call into the
native library. They still link against it, though — the crate's other code
references the C ABI — so `cargo test` builds `onnxsim_c` like any other build.
Use the prebuilt-ORT fast path (`ONNXSIM_PREBUILT_ORT=1`) to avoid compiling
ONNX Runtime from source.

### Coverage

Rust line/region coverage uses
[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov). Because the tests
link the native library, measuring coverage builds it too; the prebuilt-ORT fast
path keeps that quick:

```sh
cargo install cargo-llvm-cov          # once
rustup component add llvm-tools-preview

# Terminal summary of the workspace's Rust coverage.
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --workspace

# Accumulate the default run and the native-only integration test, then render
# an HTML report and a Cobertura XML (the format the project's other coverage
# reports use).
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --workspace --no-report
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --no-report -- --ignored list_optimizers_is_non_empty
cargo llvm-cov report --html          # target/llvm-cov/html/index.html
cargo llvm-cov report --cobertura --output-path rust-coverage.xml
```

Branch coverage needs the **nightly** toolchain — `--branch` sets
`-Zcoverage-options=branch`, which stable rejects. Add it to both the runs and
the report to populate the branch-rate column:

```sh
ONNXSIM_PREBUILT_ORT=1 cargo +nightly llvm-cov --branch --workspace
```

On stable, `cargo-llvm-cov` still reports region/line/function coverage; only the
branch column is blank.

`cargo-llvm-cov` instruments only the wrapper crates (`onnxsim`,
`onnxsim-sys`); the C++ core is measured separately by the C++ coverage job.
In CI this runs as the `rust` job in
[`.github/workflows/coverage.yml`](../.github/workflows/coverage.yml), which uses
the nightly toolchain to collect branch coverage; its Cobertura report is folded
together with the C++, Python and JS reports into a single combined coverage
summary and pull-request comment.

The integration test in `onnxsim/tests/` is ignored by default because it needs
the linked native library and an ONNX model; see the file header to enable it.

## Standalone package build test

`cargo publish --no-verify` (see [Publishing](#publishing)) means the archive
that actually ships is never compiled by the in-tree build: `cargo build` in
`rust/` always has the C++ sources, the submodules and the sibling crate next
to it, and the published crate has none of that. Anything that only breaks
there — a file missing from the package, a path dependency that stops resolving
once cargo rewrites it to a registry dependency, a build script that assumes the
onnxsim source tree is two directories up — would otherwise surface as a broken
release on crates.io.

[`scripts/test_rust_package_standalone.sh`](../scripts/test_rust_package_standalone.sh)
closes that gap. It runs `cargo package`, unpacks both `.crate` archives into a
directory **outside** this repository, patches `onnxsim`'s registry dependency
on `onnxsim-sys` back to the freshly unpacked sibling, and builds the unpacked
crates plus a throwaway downstream crate against them:

```sh
# Check-only (no native build, ~a minute): type-checks the unpacked crates and
# a consumer of them, and asserts that a build with no mode selected fails fast
# with an actionable message.
scripts/test_rust_package_standalone.sh

# Link and run the packaged crates against a native library. `--lib-dir auto`
# reuses whatever a previous `cargo build` in rust/ already produced; pass an
# explicit `dir[:dir...]` for a library built elsewhere.
scripts/test_rust_package_standalone.sh --lib-dir auto --model model.onnx

# Or build the native library from source against a checkout (slow):
ONNXSIM_PREBUILT_ORT=1 scripts/test_rust_package_standalone.sh --source-dir "$PWD"
```

Useful options: `--workdir DIR` to unpack somewhere specific (it must be outside
the repository, or the build script would find the C++ sources after all),
`--keep` to leave the unpacked tree and the generated consumer crate around for
inspection, and `--model PATH` to have the consumer actually simplify a model
once the native library is linked.

In CI this runs as the `package` job in
[`.github/workflows/rust.yml`](../.github/workflows/rust.yml) (check-only, on
every PR) and again at the end of the `build` job with `--lib-dir auto`, which
links the packaged crates against the native library that job just built and
runs them.

## Publishing

`onnxsim-sys` builds by shelling out to CMake against the onnxsim C++ source
tree that lives outside the crate directory (`../..` from
`rust/onnxsim-sys`), compiling ONNX Runtime, onnx, onnx-optimizer and
protobuf from source. `cargo publish`'s default verification step packages
the crate into an isolated directory with no monorepo around it, so that
build can't succeed there — every publish (automated or manual) has to pass
`--no-verify` to skip it. This does mean the crate isn't buildable from
crates.io metadata alone: consumers need `ONNXSIM_SOURCE_DIR` pointed at a
checkout of this monorepo (with submodules), or a pre-built library via
`ONNXSIM_LIB_DIR` — see "Building the native library" above.

### Automated (GitHub Actions)

The `publish` job in
[`.github/workflows/rust.yml`](../.github/workflows/rust.yml) runs whenever a
GitHub release is published (tag `vX.Y.Z`), after the `lint` and `build` jobs
pass. It bumps every binding manifest to the release tag
(`scripts/bump_binding_versions.sh`, the same script the `lint`/`build` jobs
run to validate against), then publishes `onnxsim-sys`, polls
`https://crates.io/api/v1/crates/onnxsim-sys/<version>` until it lands on the
index, and publishes `onnxsim`.

It authenticates via crates.io [Trusted Publishing](https://crates.io/docs/trusted-publishing)
(OIDC) rather than a long-lived API token: the job requests an `id-token`,
[`rust-lang/crates-io-auth-action`](https://github.com/rust-lang/crates-io-auth-action)
exchanges it for a short-lived (30-minute) crates.io token that's revoked when
the job ends. This requires a trusted-publishing config on crates.io for both
`onnxsim-sys` and `onnxsim` naming this repository, the `rust.yml` workflow,
and the `cargo` environment (which the job runs under, matching that config)
— no repository secret needed.

### Manual

```sh
cd rust
cargo publish -p onnxsim-sys --no-verify
# wait for crates.io's index to pick up onnxsim-sys, then:
cargo publish -p onnxsim --no-verify
```

`onnxsim-sys` must publish first and be visible on the index before
`onnxsim` (which depends on it by version) can publish.

## License

Apache-2.0, matching the parent project.
