# Pyodide / wasm32-emscripten Python extension

**Status: full `onnxsim.simplify()` runs under a real Pyodide runtime,
using onnx's real, unmodified PyPI wasm wheel.** CI
(`.github/workflows/pyodide-wasm.yml`) builds `onnxsim_cpp2py_export.abi3.so`
against **Pyodide 0.29.4** (Emscripten 4.0.9), loads it into that exact
Pyodide runtime (the `pyodide` npm package under Node), confirms the
low-level extension imports and runs, then installs `onnx` from PyPI via
`micropip` for real and runs a full `onnxsim.simplify()` call on a real
model. See `scripts/pyodide_smoke_test.mjs` and the workflow for exactly
what that covers.

**Newly added, not yet confirmed green in CI**: a real, `micropip`-
installable wheel is now built and smoke-tested as part of the release
flow (`build_wheel_pyodide` in `.github/workflows/build-and-test.yml`) --
see "Distributable wheel and release flow" below for what it does and why
it's wired in as best-effort (`continue-on-error: true`) rather than a
release-blocking step.

This resolves what earlier revisions of this doc described as a structural
gap: PyPI's only `onnx` wasm wheel is tagged for Pyodide ABI epoch
`2025_0`, and this repo's toolchain originally targeted Pyodide 314.0.5
(epoch `2026_0`) -- a real, then-unmatched epoch mismatch that made
`import onnx` fail inside Pyodide outright. The fix wasn't a change to
onnxsim's code or a protobuf downgrade (an intermediate hypothesis that
turned out to be wrong -- see "The ABI epoch mismatch, and how it was
actually resolved" below) -- it was targeting the specific Pyodide release
whose epoch already matches `onnx`'s wheel: **Pyodide 0.29.4**.

## What this is

`build_wasm_pyodide.sh` cross-compiles onnxsim's nanobind Python extension
(`onnxsim_cpp2py_export`, the module `import onnxsim` loads) for
`wasm32-emscripten`, producing `onnxsim_cpp2py_export.abi3.so` as a Pyodide
**side module** -- the format `pyodide.loadPackage` / `micropip` expect for a
compiled extension: a relocatable WebAssembly binary that Pyodide's own
runtime `dlopen`s and links against the main Pyodide module at import time,
rather than a self-contained standalone `.wasm`.

This is a **third** build path, separate from and not touching:

- **The Python wheel build** (`setup.py`) -- builds the same extension for
  the *host* platform, always with `ONNXSIM_BUILTIN_ORT=OFF`. See
  `CLAUDE.md` for why this never compiles ONNX Runtime.
- **The standalone WASM CLI build** (`build_wasm.sh`) -- builds
  `onnxsim_bin`, a self-contained CLI/JS module for the browser, with
  `ONNXSIM_PYTHON=OFF`.

Pyodide needs neither of those: it needs the *Python extension*, cross
compiled for wasm32, in Pyodide's own module format. Hence a new script
rather than a flag on either existing one.

## Why this needs a script at all

Four separate obstacles had to be worked around to get from "onnxsim builds
for the host" to a linkable wasm side module. None of them are onnxsim bugs
in the usual sense -- they're toolchain/version-matching issues specific to
cross-compiling a CPython C extension with CMake under Emscripten.

### 1. protoc/protobuf version matching

Same requirement as `build_wasm.sh`'s existing protoc handling (see that
script and `docs/wasm_ort_web.md`'s protobuf section): CMake's
`ONNX_CUSTOM_PROTOC_EXECUTABLE` needs a **host** protoc whose codegen matches
the vendored protobuf runtime being linked, or generated `.pb.*` code
mismatches the runtime headers. This script reuses `build_wasm.sh`'s exact
protoc discovery/build logic (`third_party/onnx/workflow_scripts/protobuf/build_protobuf_unix.sh`)
rather than duplicating or diverging from it.

### 2. Plumbing target Python headers through three separate CMake hint namespaces

`ONNXSIM_PYTHON=ON` triggers Python detection in onnxsim's own
`CMakeLists.txt`, onnx's, onnx-optimizer's, and nanobind's own CMake config --
and CMake's `FindPython`, `FindPython3`, and nanobind's config each read
hints from a **different** variable namespace, so missing any one of them
fails outright even with the other two correctly set:

- onnx's own `CMakeLists.txt` calls the **versioned** `find_package(Python3
  ...)`, correctly split into `Interpreter`/`Development` -- hints:
  `Python3_EXECUTABLE`, `Python3_INCLUDE_DIR(S)`.
- onnx-optimizer's `CMakeLists.txt` instead calls the **generic, unversioned**
  `find_package(Python 3 REQUIRED COMPONENTS Interpreter
  Development.Module)` -- a separate CMake module with its own variables:
  `Python_EXECUTABLE`, `Python_INCLUDE_DIR(S)`. Passing only the `Python3_*`
  hints leaves this call unable to find `Python.Development` at all
  (`Could NOT find Python (missing: Python_INCLUDE_DIRS
  Development.Module)`) -- this only surfaces on a fresh configure with no
  stale `CMakeCache.txt` from a previous, differently-argued run to paper
  over it, which is how it slipped past local testing before CI caught it.
- nanobind's own CMake config reads the undocumented plural
  `Python_INCLUDE_DIRS` variable specifically (not `Python3_INCLUDE_DIRS`).

So both the versioned and unversioned forms of every hint need to be passed,
to the same target headers, together with:

- Host and target Python **minor versions must match** (e.g. host
  `python3.14` for a Pyodide release shipping CPython 3.14.x), since
  `Python3_EXECUTABLE`/`Python_EXECUTABLE` point at the *host* interpreter
  CMake actually runs to introspect Python, while the `INCLUDE_DIR(S)` hints
  point at the *target*'s headers.
- `Python3_FIND_ABI`/`Python_FIND_ABI` set to `ANY;ANY;ANY;ANY`, to avoid a
  strict-ABI mismatch rejection between the host interpreter and the
  target's ABI tag.
- `Python3_EXECUTABLE`/`Python_EXECUTABLE` must be an **absolute path**, not
  a bare command name. A bare `python3` left CMake's own `find_program()`
  free to re-resolve it -- which, only intermittently (same toolchain
  versions, a fresh checkout/configure each time, no stale `CMakeCache.txt`
  to blame), picked a different, wrong-architecture interpreter (the CI
  runner's native `/bin/python`) for onnx-optimizer's generic
  `find_package(Python ...)` call instead, failing the configure with
  `Could NOT find Python ... Wrong architecture for the interpreter
  "/bin/python"`. `build_wasm_pyodide.sh` resolves `PYTHON_EXECUTABLE` via
  `command -v` before passing it to CMake so there is nothing left for
  CMake to re-resolve.

### 3. nanobind 3.0.0's missing `<cstdio>` include

`nb_backend.h` / `nb_types.h` / `ndarray.h` use `stderr`/`fprintf` without
including `<cstdio>`. This works by accident on glibc hosts, which pull it in
transitively through some other header, and fails under Emscripten's libc++,
which doesn't. This is a real upstream nanobind bug worth reporting there --
not an onnxsim-specific issue. Worked around here with `-include cstdio` on
`CMAKE_CXX_FLAGS` for this build only.

### 4. Emscripten's CMake toolchain disables native MODULE/SHARED support

Emscripten's own CMake toolchain file
(`$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`)
hardcodes `set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)`.
By design, CMake cannot produce a `MODULE`/`SHARED` library under this
toolchain -- the `onnxsim_cpp2py_export` target (a nanobind extension module,
normally `MODULE`) silently falls back to being archived as a **static**
library containing only its own single object file
(`cpp2py_export.cc.o`). This is also why Pyodide's own `pyodide build`
bypasses CMake/setuptools' dynamic-linking support entirely, via a
compiler-wrapping shim (`pywasmcross`), instead of relying on it.

The fix -- and the reason this script exists rather than just being a CMake
invocation -- is a **manual link step outside CMake**, after the CMake build
completes, using the exact object file and dependency archives CMake already
produced:

```sh
em++ -O2 -sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS=_PyInit_onnxsim_cpp2py_export \
  -o onnxsim_cpp2py_export.abi3.so \
  CMakeFiles/onnxsim_cpp2py_export.dir/onnxsim/cpp2py_export.cc.o \
  <every *.a found under the build dir: abseil, protobuf, onnx,
   onnx-optimizer, onnxsim's own static libs, nanobind-static-abi3, ...>
```

`-sEXPORTED_FUNCTIONS=_PyInit_onnxsim_cpp2py_export` is required: without an
explicit root symbol, the linker's dead-code elimination strips the module
down to a couple KB, since nothing in this standalone link otherwise
references the nanobind entry point (there's no `main()` or other symbol
pulling it in, unlike inside Pyodide's real loader). The script discovers the
exact entry symbol from the object file via `llvm-nm` rather than assuming it
matches the module name, and discovers the archive list by scanning the build
directory for `*.a` at run time -- **not** a hand-maintained list, which
would silently go stale as dependencies change.

## Building

```sh
# Activate an emsdk whose Emscripten version compiles onnxsim's vendored
# protobuf cleanly AND matches onnx's published wasm wheel's ABI epoch.
# pyodide-build's own default (Emscripten 3.1.46 / clang-18) fails on a
# protobuf `constinit` compile error; Emscripten 4.0.9, matching Pyodide
# release 0.29.4 (ABI epoch 2025_0, the same epoch onnx's wheel is tagged
# for), is the recommended target -- see "The ABI epoch mismatch" below for
# why the epoch match matters, not just "new enough to compile". Use the
# SAME emsdk for the whole run -- see the script's own header comment for
# why.
source /path/to/matching/emsdk/emsdk_env.sh

PYODIDE_PYTHON_INCLUDE=/path/to/xbuildenv/.../include/python3.13 \
PYTHON_EXECUTABLE=/path/to/host/python3.13 \
  ./build_wasm_pyodide.sh
```

`PYTHON_EXECUTABLE` needs `nanobind` pip-installed (`pip install nanobind`);
CMake shells out to `python -m nanobind --cmake_dir` on it to locate
nanobind's CMake package. `PYODIDE_PYTHON_INCLUDE` is the target Pyodide
release's Python headers dir, e.g. from a Pyodide xbuildenv
(`pyodide xbuildenv install <version>`, or reuse one already downloaded to
`~/.cache/pyodide-build`). Both are asserted with a clear error if unset. See
the script's own header comment for the full prerequisite list and defaults
(`BUILD_DIR` defaults to `build-wasm-pyodide`, reused across reruns like
`build_wasm.sh`'s `BUILD_DIR`).

Output: `<BUILD_DIR>/onnxsim_cpp2py_export.abi3.so`, the script's last line.

## The ABI epoch mismatch, and how it was actually resolved

`onnx`'s only PyPI wasm wheel
(`onnx-1.22.0-cp312-abi3-pyemscripten_2025_0_wasm32.whl`) is tagged for
Pyodide's ABI epoch `2025_0`. This repo's wasm build originally targeted
Pyodide 314.0.5 (epoch `2026_0`), so `micropip.install("onnx")` failed
outright:

```
ValueError: Wheel was built with Emscripten vpyemscripten.2025.0 but
Pyodide was built with Emscripten v5.0.3
```

**First hypothesis, checked and wrong**: that onnx's wheel avoided the
protobuf `constinit` compile error (obstacle 1 below) by building with
`ONNX_USE_LITE_PROTO=ON` (onnx's own standard release convention, confirmed
in `.github/workflows/release_pyodide_cibw.yml` upstream), so downgrading
onnxsim's own protobuf version below the one that introduced that pattern
(v30.0) might let onnxsim compile under the same old toolchain. Tested
directly: it does NOT help -- the offending code
(`fixed_address_empty_string` in `port.cc`) is compiled into
`libprotobuf-lite` too, not just full `libprotobuf`; lite vs full doesn't
matter for this specific bug.

**What actually explains it**: Pyodide's ABI epoch tracks the *CPython*
version, not the Emscripten version. Checked directly in Pyodide's own git
history (`Makefile.envs`, the commit that bumped `PYODIDE_ABI_VERSION`
`2025_0` -> `2026_0`): `PYODIDE_EMSCRIPTEN_VERSION` stayed at the same
`5.0.x` line across that epoch boundary -- only `PYVERSION` changed (3.13.2
-> 3.14.2). Epoch `2025_0` itself, per the last Pyodide release that shipped
it (**0.29.4**), pairs with **Emscripten 4.0.9** -- a modern-enough
toolchain that (like 5.0.3) has no trouble with onnxsim's protobuf 31.1 at
all, full (non-lite) proto included. So the fix is simply: target Pyodide
0.29.4 instead of 314.0.5. No protobuf downgrade, no upstream `onnx`
change needed.

**Verified, concretely**, before wiring this into CI: built
`onnxsim_cpp2py_export.abi3.so` against Pyodide 0.29.4's xbuildenv with
unmodified protobuf 31.1 -- compiled clean. Downloaded onnx's real
`onnx-1.22.0-*-pyemscripten_2025_0_wasm32.whl` from PyPI and loaded its
compiled extension (`onnx_cpp2py_export.cpython-313-wasm32-emscripten.so`)
directly via `importlib` inside a real Pyodide 0.29.4 runtime (`pyodide`
npm package under Node) -- it initialized successfully, side by side with
onnxsim's own extension in the same runtime. `scripts/pyodide_smoke_test.mjs`
now does the equivalent through the normal `micropip.install("onnx")` path
(which additionally resolves `numpy`/`protobuf`/`ml_dtypes` from Pyodide's
own package repository automatically) and runs a full `onnxsim.simplify()`
call as its final check.

## Browser demo

`scripts/pyodide_demo/index.html` is a self-contained, client-side-only demo
page: drop in a real `.onnx` file, either run the full `onnxsim.simplify()`
pipeline (installs `onnx` from PyPI via `micropip`, same as the CI smoke
test) or one of onnxsim's native quantization/precision-conversion passes
(weight-only int8/int4/int16/block, dynamic, ternary, bf16/fp16/fp8) that
work without `onnx` present at all -- see real before/after size and
op-count deltas, download the result. Everything runs in the browser via
Pyodide 0.29.4 -- no server involved.

**To try it:**

```sh
# 1. Build the extension (see "Building" above), or download a recent build
#    from the "Upload the built module as a workflow artifact" step of a
#    pyodide-wasm.yml CI run instead of building locally.
# 2. Put it next to the demo page:
cp <BUILD_DIR>/onnxsim_cpp2py_export.abi3.so scripts/pyodide_demo/
# 3. Serve the directory (opening the file directly won't work -- the page
#    fetches the .so over HTTP):
python3 -m http.server -d scripts/pyodide_demo 8000
# 4. Open http://localhost:8000/ and click "Load runtime".
```

The native quantization passes remain available (and remain the faster
option -- no `onnx`/`numpy` download needed) since several bindings in
`onnxsim/cpp2py_export.cc` (`_list_optimizers`, `_model_metrics`, and the
`quantize_*` passes) operate directly on raw serialized ONNX `ModelProto`
bytes using onnxsim's own *statically-linked* copy of onnx/onnx-optimizer/
protobuf, needing no Python `onnx` package at all.

**Hosting**: the page fetches `./onnxsim_cpp2py_export.abi3.so` as a
same-directory relative path. That `.so` is not committed (like every
other `*.so` in this repo, it's gitignored) -- get one via the CI artifact
or a local build, as above. `static.yml` stages this demo (both the page
and a freshly-built `.so`, imported as a cross-workflow artifact from
`pyodide-wasm.yml`) at `pyodide-demo/` alongside its existing ORT-web/JS
convertmodel deploy, on every production deploy and `/preview` -- see its
"Pyodide demo" step for exactly how, including the fallback it uses when
the exact commit being deployed has no matching `pyodide-wasm.yml` run of
its own (most commits don't, since that workflow is path-filtered).

## Distributable wheel and release flow

`build_wasm_pyodide.sh` produces a loose `.so`, not something `micropip`
can install directly. Closing that gap needed two things: (1) `setup.py`'s
`build_ext` step doing the same manual `em++ -sSIDE_MODULE=2` relink
`build_wasm_pyodide.sh` does (CMake alone still can't produce a real side
module -- see above), and (2) assembling that relinked `.so` plus onnxsim's
pure-Python files into an actual wheel with the right Pyodide platform tag.

**(1)**, `setup.py`'s `ONNXSIM_WASM_SIDE_MODULE_RELINK` flag, is gated
behind an explicit opt-in env var so it has zero effect on the normal
native wheel build:

```sh
ONNXSIM_WASM_SIDE_MODULE_RELINK=1 <the rest of a normal `pip wheel .` /
  `python setup.py build_ext` invocation, run under an activated emsdk>
```

When set, `build_ext` performs the exact relink `build_wasm_pyodide.sh`
does (same object-file/archive discovery, same `-sEXPORTED_FUNCTIONS`
entry-symbol handling) on CMake's output before copying it into the wheel,
so whatever packages the wheel afterward picks up a genuine dynamic module
instead of a static archive. Unset (the default), this code path is never
even evaluated.

**(2)** is now automated: `.github/workflows/build-and-test.yml`'s
`build_wheel_pyodide` job runs `pyodide build` (pyodide-build's own
`pywasmcross` compiler-wrapping) with `ONNXSIM_WASM_SIDE_MODULE_RELINK=1`
set, against the same pinned toolchain as `pyodide-wasm.yml` (Emscripten
4.0.9 / Pyodide 0.29.4, for the same ABI-epoch reason as everywhere else in
this doc). `pyodide build`'s own compiler wrapping does NOT solve the
`SIDE_MODULE` problem by itself (confirmed directly: it wraps individual
`em++`/`emcc` invocations, but never CMake's own generator-level choice of
link rule, which is what's actually blocked by
`TARGET_SUPPORTS_SHARED_LIBS=FALSE`) -- that's exactly the gap (1) closes,
so the two combined are what let `pyodide build` produce a real,
`micropip`-installable wheel (correct `.dist-info`/`RECORD` and platform
tag) instead of a broken one containing a static archive.

The job verifies the actual wheel it produces, not just the loose `.so`:
`scripts/pyodide_wheel_smoke_test.mjs` writes it into a real Pyodide
0.29.4 runtime's virtual filesystem, `micropip.install("emfs:...")`s it
(with the default `deps=True` -- `onnxsim`'s pure-Python code imports
`numpy`/`onnx` unconditionally at module level, so even a plain `import
onnxsim` needs them resolved, not just `onnxsim.simplify()`; this also
exercises the wheel's own declared dependency metadata, not just the
extension), imports the *installed* `onnxsim` from its real site-packages
path, and calls `_list_optimizers()` to confirm it's genuinely
functional.

**Wired into the release flow, but as a best-effort addition, not a
blocking one**: `build_wheel_pyodide` runs on `push`/`release`/`schedule`/
`workflow_dispatch` (not a plain `pull_request` -- the underlying wasm32
build is already functionally validated per-PR by the separate,
path-filtered `pyodide-wasm.yml`; this job's own job is producing the
release artifact, which has no consequence until a tag exists), and
uploads its wheel as `python-dist-pyodide` -- the same `python-dist-*`
naming convention `upload_pypi` already downloads by wildcard, so no
change was needed there beyond adding the job to `needs:` for ordering.
It runs with `continue-on-error: true`: this is genuinely new, first-time-
in-CI automation (previously validated only once, by hand, at a now-
outdated ABI epoch -- 314.0.5/`2026_0`, before the epoch-matching fix
above), so a failure here must not hold back the native wheels every
existing user actually depends on. The daily schedule already exercises
the whole thing for real, safely, before it ever matters: `upload_pypi`
publishes scheduled (and manually dispatched) builds to Test PyPI, not the
real index, so this job's first few real runs are a genuine dry run, not a
live release gate.

**Still an open question, not yet observed**: this is the first time this
exact combination (`pyodide build` + the `ONNXSIM_WASM_SIDE_MODULE_RELINK`
relink, driven by CI rather than by hand) has actually been run end to
end. The previous validation of this general approach (before this job
existed) hand-assembled a wheel's `.dist-info` instead of using `pyodide
build`, and did so at Pyodide 314.0.5/epoch `2026_0`, not the current
0.29.4/`2025_0` pin -- so whether `pyodide build` itself runs cleanly
against onnxsim's CMake-based `setup.py` (in particular, whether its
cross-build environment's `sysconfig` shim correctly redirects
`Python_INCLUDE_DIR`/`Python_EXECUTABLE` to the wasm32 target the way it's
expected to for CMake-based extensions) is confirmed only once this job
has actually gone green in CI. If it needs further iteration, that's
expected -- see this file's own history for how many real, CI-discovered
fixes the rest of this pipeline needed before it worked end to end.
