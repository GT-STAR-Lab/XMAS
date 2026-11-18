# XMAS

Graph-based explanations for multi-agent coordination policies, with
attention-entropy regularization during PPO training and post-hoc evaluation
using AttentionExplainer, GNNExplainer and GraphMask.

This source release contains the captured primary training, evaluation and
plotting workflows for the [paper](45426_Evaluating_and_Improving.pdf) and
[technical supplement](45426_Evaluating_and_Improving_Technical%20Supplement.pdf),
their shared model/observation code, and the dependency edits needed to run them.
See the [source map](docs/PAPER_SOURCE_MAP.md) for the retained components.

## Installation

Use Python 3.11. The captured GPU environment used PyTorch 2.1.1+cu121;
compiled PyG extensions must match your Torch/CUDA platform. See the
[dependency guide](reproducibility/README.md) for build details and restoration.

```bash
python -m pip install -r reproducibility/requirements-runtime.txt
python -m pip install --no-deps -e VectorizedMultiAgentSimulator
python -m pip install --no-deps -e .
python reproducibility/restore_dependency_edits.py --apply
python reproducibility/restore_dependency_edits.py --check
```

VMAS is vendored as a complete runtime package, including its modified tasks,
shared simulator, standard scenarios and rendering data. Its license and the
captured dependency license notices are retained. The custom TorchRL and PyG
sources are required for the attention loss, explainers, metrics and visualization.

## Training and evaluation

| Task | Training entrypoint | Evaluation entrypoint |
|---|---|---|
| Navigation | `examples/multiagent_gnn_ppo_nav_direct_gnn_action.py` | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_only_load_and_eval.py` |
| Passage | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_passage.py` | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_only_load_and_eval_passage.py` |
| Discovery | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_discovery.py` | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_only_load_and_eval_discovery.py` |

Run each experiment in a fresh directory. The scripts prompt for team size
and other task options, and read/write checkpoints in their working directory:

```bash
XMAS_ROOT="$PWD"
mkdir -p experiments/navigation
cd experiments/navigation
python "$XMAS_ROOT/examples/multiagent_gnn_ppo_nav_direct_gnn_action.py"
# After training, evaluate model.pth from this same directory:
python "$XMAS_ROOT/examples/multiagent_gnn_ppo_nav_direct_gnn_action_only_load_and_eval.py"
```

`examples/parse_and_create_boxplots.py` is the original explanation plotting
helper. Run it from a task/team-size directory containing the two condition
folders `no_attn_ent/` and `attn_ent/`. The passage coefficient-ablation plotting
helper is `examples/parse_and_create_boxplots_passage_ablation.py`.

The numerical training, masking and metric implementations are preserved.
[Implementation notes](docs/IMPLEMENTATION_NOTES.md) document the navigation
coefficient discrepancy and interpretation of the original statistical outputs.
Rendering may require a suitable display/OpenGL setup.

## Scope and provenance

The shared model code originates from
[Het-MAMP](https://github.com/GT-STAR-Lab/het-mamp); the modified simulator
originates from [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).
The [source manifest](docs/SOURCE_MANIFEST.json) records the retained source hashes.

Exploratory variants, generated output collections, migration tooling,
post-manuscript training runs and reviewer reanalyses are maintained separately.
Their complete original records have been preserved. The inclusion rule is the
paper workflow and its runtime dependencies, independent of experiment outcome.

Model checkpoints and complete raw metric CSV collections are separate research
artifacts. This repository makes no additional claim of robustness across
training seeds or exact recovery of manuscript tables from a fresh training run.
Dependency source checks alone do not validate a complete fresh-environment replay.
