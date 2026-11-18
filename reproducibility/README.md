# Captured runtime dependencies

The paper-era implementation uses local edits to TorchRL 0.2.0 and PyTorch
Geometric 2.4.0, plus the modified VMAS source vendored at
`VectorizedMultiAgentSimulator/`. This directory supplies the captured patches,
source snapshots, upstream licenses and a restoration helper. It documents the
runtime needed by the retained implementation; it contains no additional
training-seed evaluation results.

## Installation and restoration

Use a fresh Python 3.11 environment:

```bash
python -m pip install -r reproducibility/requirements-runtime.txt
python -m pip install --no-deps -e VectorizedMultiAgentSimulator
python -m pip install --no-deps -e .
python reproducibility/restore_dependency_edits.py --apply
python reproducibility/restore_dependency_edits.py --check
```

Run restoration before importing the affected libraries into a notebook or
long-running process. The helper checks package versions and source hashes
before writing, refuses unrecognized contents, and verifies every restored
file. `--check` performs no writes. Its default selection is TorchRL and PyG.

The captured environment used PyTorch `2.1.1+cu121` and these native extensions:

```text
pyg-lib==0.3.1+pt21cu121
torch-cluster==1.6.3+pt21cu121
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
torch-spline-conv==1.2.2+pt21cu121
```

Select compatible native wheels for your Torch/CUDA platform. The requirements
file records observed direct dependency versions; it is not a fully tested
transitive/native environment lock. Exact before/after source hashes are retained in the patch manifest.

The original plotting helper additionally requires `statannot==0.2.3`. To
restore its optional legacy debug-print edit as well:

```bash
python -m pip install statannot==0.2.3
python reproducibility/restore_dependency_edits.py --apply --packages torchrl torch-geometric statannot
```

## Captured source edits

| Package | Captured changes |
|---|---|
| TorchRL | PPO attention-entropy `loss_attn`; exploration variant; VMAS wrapper comments/import |
| PyG | Custom AttentionExplainer attention access; GraphMask changes; regression fidelity; graph visualization API; GNNExplainer comments |
| statannot | Optional debug print for legacy plotting |

The custom PPO loss, AttentionExplainer, GraphMask and fidelity sources affect
the primary numerical workflow. The visualization changes are also needed by
scripts that pass `env=` to `visualize_graph`. Preserve the entire captured
TorchRL/PyG bundle when running those scripts unchanged.

The [patch manifest](dependency_audit_20261117/patch_manifest.json) records exact
before/after hashes. [Readable diffs](dependency_audit_20261117/patches) and
captured sources are accompanied by upstream licenses. The restoration helper
uses relative bundle paths and does not require the original machine's absolute
installation paths. The vendored VMAS tree contains the modified task and
simulator sources and rendering data; no submodule checkout is required.

## Validation scope

The captured changes were compared with pristine wheels at the recorded
versions. Patch application and the restoration helper reproduced the captured
files byte for byte, including refusal of unexpected source contents. Those
checks validate source restoration. A complete fresh-environment training and
evaluation replay remains unverified.

The bundle retains complete patched distribution files, corresponding upstream
sources, diffs and license notices. Its extra captured library methods do not
change which paper entrypoints are provided. See the root README and
[implementation notes](../docs/IMPLEMENTATION_NOTES.md) for release scope.
