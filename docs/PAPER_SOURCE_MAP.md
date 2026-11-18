# Source map for the paper workflows

The retained primary entrypoints share the model architectures and task paths
used for the paper's three environments. Their source bytes are preserved;
this map does not recover undocumented historical checkpoint-selection rules
or every command used to produce the manuscript figures.

| Component | Retained source | Role |
|---|---|---|
| Navigation | `examples/multiagent_gnn_ppo_nav_direct_gnn_action.py` and its `*_only_load_and_eval.py` counterpart | Train and explain navigation policies |
| Passage | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_passage.py` and `*_only_load_and_eval_passage.py` | Train and explain passage policies |
| Discovery | `examples/multiagent_gnn_ppo_nav_direct_gnn_action_discovery.py` and `*_only_load_and_eval_discovery.py` | Train and explain discovery policies |
| Main plotting | `examples/parse_and_create_boxplots.py` | Read original metric CSVs and generate explanation comparisons |
| Passage coefficient ablation | `examples/parse_and_create_boxplots_passage_ablation.py` | Plot coefficient-ablation data related to supplement Appendix L |
| Shared model | `hetmamp/modules/multiagent_models.py`, `hetmamp/modules/gnn.py` | Shared encoder/GNN/decoder policy and attention/message passing |
| Observations and communication | `hetmamp/envs/transforms.py` | Observation transforms and communication graph construction |
| Environments | `VectorizedMultiAgentSimulator/vmas/` | Modified navigation, passage and discovery tasks plus the complete VMAS runtime |
| Custom dependencies | `reproducibility/` | Restore the captured PPO attention loss, explainer, fidelity and visualization edits |

Supplement Appendices M and N describe task architectures and training settings.
The supplementary intermediate-checkpoint, team-size and coefficient studies
reuse these task/model/evaluation components. Historical checkpoint collections
and complete figure-generation invocations are separate artifacts. The navigation
evaluator constructs the shared policy for the requested team size; loading
compatible shared weights supports evaluation at a different team size.

Unused upstream Hydra/algorithm entrypoints, exploratory GNN variants and their
output folders are excluded. Complete dependency files remain intact to preserve
the captured APIs, so the vendored VMAS package also contains standard scenarios
and helpers beyond the three tasks used here.

See [implementation notes](IMPLEMENTATION_NOTES.md) for known discrepancies and
[the dependency guide](../reproducibility/README.md) for environment setup.
