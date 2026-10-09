# Versioning ONNX graphs as text

This workflow keeps an ONNX graph in git as readable text, verifies every change
against the original model on test inputs, and bisects a failing change to the
fused block that caused it. Weights never enter git. They stay in the original
`.onnx` file and any snapshots you keep, and the text refers to them by hash.

The entry points are:

- `python -m onnxsim.versioning_cli`: the command line, used for everyday work.
- `onnxsim.versioning_project`: the same operations as Python functions.
- `onnxsim.versioning`: the lower-level pieces (text codec, digests, equivalence
  check, bisect, manifest).

## Project layout

A project is a directory you commit to git:

```
proj/
  model.txt        canonical graph text; initializers are references, not values
  manifest.json    base file and digest, plus one entry per verified change
```

The original model and any weight files are kept outside git, and you are
responsible for keeping them. The manifest records the original file's path
relative to the project and its digest, so a changed original is detected.

`model.txt` is ONNX text that `onnx.parser` accepts. Each initializer is a
marker comment, which the parser ignores:

```
# onnxsim-init {"digest": "sha256:9ac1…", "dims": [4, 4], "elem_type": "FLOAT", "name": "W"}
<ir_version: 10, opset_import: ["" : 21]>
g (float[1,4] x) => (float[1,4] y)
{
   h = MatMul (x, W)
   y = Relu (h)
}
```

Edit `model.txt` by hand or with a tool, and commit it. Git diffs show graph
changes; a changed weight shows up only as a changed digest.

## Workflow

1. **Start from the original model.**

   ```
   python -m onnxsim.versioning_cli init model.onnx proj
   ```

   This writes `proj/model.txt` and `proj/manifest.json`. It fails if the
   directory already holds a project.

2. **Change the graph.** Edit `proj/model.txt`. Add or remove nodes, change
   attributes, or point a node at a different reference. A weight can be
   referenced by the digest of another tensor that exists in some `.onnx` file
   you supply.

3. **Build to check it loads.**

   ```
   python -m onnxsim.versioning_cli build proj -o out.onnx -w other_weights.onnx
   ```

   `build` resolves every digest in `model.txt`, looking in the original file
   and in each `-w` file. A digest that no file provides is an error, and the
   output is a standard `.onnx` file you can run anywhere.

4. **Verify against the original.**

   ```
   python -m onnxsim.versioning_cli verify proj --cases cases.json --label "fold scale"
   ```

   This runs the original and the edited graph on every test case and compares
   their outputs. It appends a step to `manifest.json`, whether it passes or
   fails.

5. **If it fails, read the culprit.** On failure, `verify` bisects the change and
   prints the fused block responsible:

   ```
   verdict: fail
   culprit: unit 1 (fusion=default)
     blocks:       MatMul+Relu
     nodes:        Relu:y, Sigmoid:y
     initializers: -
   ```

   The step in the manifest keeps the same culprit data.

6. **Commit** `model.txt` and `manifest.json` together with the change.

To write a simplified model, add `--simplify`, and pass any keyword option of
`onnxsim.simplify` with `--simplify-opt NAME=VALUE` (repeatable; it implies
`--simplify`):

```
python -m onnxsim.versioning_cli build proj -o out.onnx \
    --simplify-opt skip_fuse_bn=true --simplify-opt tensor_size_threshold=4KB
```

Each option name is checked against `onnxsim.simplify`'s signature before anything
runs, so a typo is an error and nothing is written. A value is read as JSON, so
`true`, `4` and `["a"]` keep their types; anything that isn't valid JSON is taken
as a string. If simplify's own check fails, `build` exits 1.

Every build is appended to `builds` in `manifest.json`, with the output's file
digest, the source graph hash, the simplify options (`null` when simplify was not
used), whether simplify's check passed, the onnxsim version, and the executor. The
simplified file differs from the graph text, so the manifest is the record of how
it was produced.

`verify` checks the unsimplified graph. Simplify changes the output, so a simplified
build is not verified unless you verify it separately.

`status` summarizes a project: the base file, the base graph hash, the current
graph hash, and each recorded step with its verdict.

## Test cases

`verify` reads its test cases from a JSON file:

```json
{
  "cases": [
    {
      "name": "random-smoke",
      "kind": "generated",
      "seed": 20261009,
      "specs": [
        {"name": "x", "dtype": "float32", "shape": [1, 4], "low": 0.0, "high": 1.0}
      ],
      "digests": {"x": "sha256:4e0a…"}
    },
    {
      "name": "recorded-sample",
      "kind": "supplied",
      "path": "samples/sample-17.npz",
      "digests": {"x": "sha256:9f3c…"}
    }
  ]
}
```

- **Generated** inputs are drawn from a seeded PCG64 generator in the order
  `specs` lists them. Each draw is repeatable, and `digests` (optional) must
  match the regenerated bytes. A mismatch usually means the NumPy version changed
  the stream, or the spec changed; the run stops with an error rather than comparing
  different inputs.
- **Supplied** inputs are read from an `.npz` file. A relative `path` resolves
  next to the cases file. Each tensor's digest is checked when the file is read.

- **ONNX test data** reads a case from the ONNX backend test layout: a
  directory with `test_data_set_<n>/input_<i>.pb` files (TensorProto), matched to
  the model's non-initializer inputs in order:

  ```json
  {"name": "add-broadcast", "kind": "onnx_test_data",
   "directory": "third_party/onnx/onnx/backend/test/data/pytorch-operator/test_operator_add_broadcast",
   "test_set": 0}
  ```

  The `output_<i>.pb` files, when present, are the official expected outputs. The
  verifier doesn't compare against them; `examples/versioning/onnx_test_data.py`
  does, using ONNX's reference evaluator.

A case may mix kinds. Cases are recorded in the manifest with their
inputs' digests, so the step's test set is a fixed identity.

## Verdicts and exit codes

Each case gets a status:

| status    | meaning |
|-----------|---------|
| `pass`    | every output matched within tolerance |
| `partial` | every judged output matched, but some outputs were not judged (see [custom backends](#custom-backends-and-partial-evaluation)) |
| `fail`    | a judged output differed beyond tolerance |

The step's verdict is `fail` if any case failed, `partial` if none failed but
some were partial, and `pass` otherwise.

`verify` exits with:

- `0` for `pass` and `partial`,
- `1` for `fail`,
- `2` for a usage or data error, such as a missing cases file, an original that
  no longer matches its recorded digest, or a digest that no weight file provides.

Tolerances default to `--atol 1e-5 --rtol 1e-4`. An output matches when every
element satisfies `|candidate − base| ≤ atol + rtol·|base|`, and its shape must
equal the base's shape. Output names must match the original's.

Use `--no-record` to check without appending a step to the manifest.

## Fusion presets

A failure is reported at the level of fused blocks, because an accelerator runs
a block as one kernel. A head op starts a block, and a tail op joins it when it
consumes the block's output and nothing else does:

- heads: `Conv`, `ConvTranspose`, `MatMul`, `Gemm`
- tails: `Relu`, `Clip`, `Sigmoid`, `Tanh`, `HardSigmoid`, `HardSwish`,
  `LeakyRelu`, `Elu`, `Gelu`, `Add`, `Mul`, `BatchNormalization`

Choose the preset with `--fusion`:

| preset       | blocks |
|--------------|--------|
| `default`    | conv and matmul heads, with the tails above (the default) |
| `conv-act`   | conv heads only |
| `matmul-act` | matmul and gemm heads only |
| `node`       | one block per node |

A change to any node of a block is one unit, and applying that unit swaps in
every node of the block from the edited graph, since the block runs as one
kernel. Units are reported with their block labels, such as `MatMul+Relu`.

## Custom backends and partial evaluation

Some backends can't run every op. By default, `verify` runs the original and
the edited graph through ONNX Runtime, and a backend failure stops the run. To
evaluate the rest of the graph on a backend that can't run some ops, pass a
backend:

```
python -m onnxsim.versioning_cli verify proj --cases cases.json \
    --backend mypkg.backend:run_node
```

`MODULE:FUNCTION` names a Python function with this signature:

```python
def run_node(node, inputs):
    """Run one ONNX node. inputs has one array per node input (None if omitted).

    Return a list of output arrays, one per node output. Raise any exception
    for an op this backend cannot run.
    """
```

The candidate runs node by node on this function. When it raises for a node:

1. The node's outputs are filled with seeded random values, shaped and typed
   from the original model's reference run. The seed is the case's seed for
   generated cases, and 0 for supplied ones, so reruns match.
2. Evaluation continues with the remaining nodes.
3. Anything computed from a filled value is marked, and so is each graph output
   that depends on one. Those outputs are not judged. The case is reported as
   `partial` and lists the outputs it skipped and the ops that failed.

A judged output that differs still fails the case. The original model always runs
on ONNX Runtime, so only the edited graph depends on your backend.

If a failed node's output name doesn't exist in the original model, its values
can't be sized, so the node's consumers are skipped.

## Bisecting a failure

On failure, bisect uses the same cases and backend to find the culprit:

- The changes are grouped into units. A unit is a connected group of modified
  nodes and initializers, and changes in the same fusion block always share a
  unit.
- Units are applied to the original one at a time, in order. Binary search finds
  the first prefix that fails, and its last unit is the culprit.
- An intermediate graph that isn't valid, for example one that reads a tensor
  nothing produces, stops the bisect with an error.

The search assumes that once a prefix fails, longer prefixes also fail. A change
whose failure depends on another change can point at the wrong unit. Check the
culprit against the diff in `model.txt` before acting on it.

## Manifest

`manifest.json` holds:

- `base`: the original file's path, its digest, and its graph hash;
- `steps`: one entry per `verify` run, in order. Each records `label`, `command`,
  `base_graph` and `output_graph` (the graph hashes before and after), `executor`
  (for example `onnxruntime 1.29.0`), `test_set`, `cases`, `verdict`, a `reports`
  list with one entry per case, and `culprit` when the verdict is `fail`.

Steps are only appended. Never edit them by hand. Each `base_graph` matches the
previous step's `output_graph`, so the steps form a chain from the original.

## Running ONNX's own test cases

`examples/versioning/onnx_test_data.py` runs every case in the ONNX submodule's
backend test data through the same checks. For each case it verifies that ONNX's
reference evaluator reproduces the official outputs, and that the model written as
graph text and rebuilt gives the same results as the original. Run it from the
repository root:

```
PYTHONPATH=. python examples/versioning/onnx_test_data.py --group simple --limit 20
```

Cases that ONNX Runtime can't run (for example opset-6 models in ONNX Runtime 1.29)
are reported as skipped with the reason, not as failures. The script exits 1 only
when an official output differs or a round trip fails.

## Python API

Everything the CLI does is available from `onnxsim.versioning_project`:

```python
from onnxsim.versioning_project import init_project, build_project, verify_project, load_cases

init_project("model.onnx", "proj")
cases = load_cases("cases.json")
result = verify_project("proj", cases, label="fold scale", backend=run_node)
print(result.passed, [r.status for r in result.reports], result.culprit)
```

`verify_project` returns a `VerifyResult` with `passed`, the per-case `reports`,
the `culprit` (or `None`), and the manifest `entry` (or `None` with `record=False`).

## Limitations

- Only top-level initializers are referenced by digest. Initializers inside
  subgraphs and local functions are printed as they are.
- The graph text is printed by `onnx.printer`. Metadata that the printer or the
  parser doesn't round-trip is not kept.
- Generated inputs depend on the NumPy version. A digest mismatch is reported,
  not worked around.
- Bisect assumes a failure only grows with more changes, so the culprit is a
  best estimate on non-monotone changes.
- Remote or profiled execution isn't wired in yet. See the RPC notes in
  [rpc.md](rpc.md) for the existing remote transport.
