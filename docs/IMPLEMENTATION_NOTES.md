# Captured implementation and interpretation

The source in this release preserves the captured paper-era implementation.
These notes identify behavior needed to interpret or run that code. They do
not report follow-up experiment outcomes or establish a new multi-seed result.

## Attention regularization

The captured TorchRL PPO patch exposes `loss_attn`. For the attention tensors
used by the task scripts, its implementation averages the categorical attention
entropy and divides again by the leading minibatch dimension. The coefficient
therefore multiplies that implemented loss, not an unnormalized entropy sum.
The exact patch is retained in the dependency bundle.

The supplement reports attention coefficients 10 for navigation and 50 for
passage and discovery. The current passage and discovery training entrypoints
multiply `loss_attn` by 50. The captured navigation entrypoint instead sets
`attn_ent_weight = 0.5` and uses:

```python
loss_vals["loss_attn"] * (
    attn_ent_weight if n_agents > 5 or attn_ent_weight == 0 else 1
)
```

Thus its effective coefficient is 1 for teams of three to five agents and
0.5 for larger teams with the captured default. Setting the variable to zero
disables that term. The retained navigation source has not been changed to
resolve this discrepancy. The historical evaluated checkpoints do not establish
the coefficient or checkpoint-selection rule used to produce them. A fresh
training run is not an established exact recreation of the published policies.

## Original evaluation and statistics

The original evaluators load one checkpoint per invocation and set 50 rollouts
per explainer. Attention and explanation calculations use the captured model,
TorchRL and PyG implementations, including the original masking and metric
behavior. The release does not substitute a corrected evaluator.

The original plotting script reads flattened timestep values from metric CSVs.
Its plotted metric error bars use standard deviations of those values, and its
active Mann–Whitney call operates on the pooled timestep rows. These outputs
do not quantify variability over independently trained policies; timesteps in
a rollout are repeated measurements. Historical CSVs alone do not verify a
mapping from rollouts to distinct training seeds. Claims about training-seed
robustness require separate run records and an appropriate analysis.

## Artifact availability

The paper and supplement are included, together with the implementation and
captured dependency edits. Model checkpoints and complete raw metric CSV
collections are separate artifacts. Source preservation and dependency hash
checks alone do not establish exact recovery of the manuscript tables or
robustness to new training seeds.
