# Machine learning

Trainable models over partitioned SMF experiments: contracts, data plane, training backends,
evaluation, interpretability, and immutable artifacts.

```{toctree}
:maxdepth: 1

architecture
quickstart
plan_reference
splits_and_masks
interpretability
artifacts_and_trust
performance
```

Training something for the first time, start with the [quick start](quickstart.md).
Wanting the design rationale, start with [architecture and ownership](architecture.md). Sizing a run, or
wondering why a read was refused, start with [performance and limits](performance.md).

Working out why a split, balancing choice, or mask behaves as it does, see
[splits, balancing, and masks](splits_and_masks.md). Choosing an attribution method, see
[interpretability](interpretability.md). Loading a model someone else produced, see
[artifacts, provenance, and trust](artifacts_and_trust.md).

Migrating from the legacy `analysis.compute.ml_*` entry points is covered by the
[ML migration guide](../tutorials/ml_migration.md).
