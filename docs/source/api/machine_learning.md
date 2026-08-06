# Machine learning API

Reference for `smftools.machine_learning`. For orientation start with the
[architecture guide](../ml/architecture.md); for a worked path start with the
[quick start](../ml/quickstart.md).

## Contracts and plans

Schemas, plans, manifests, workspace resolution, and row selection.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.contracts
   smftools.machine_learning.manifests
   smftools.machine_learning.plan
   smftools.machine_learning.selection
   smftools.machine_learning.splitting
   smftools.machine_learning.workspace
```

## Data plane

Partition reads, fitted transforms, and train-only balancing.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.data.balancing
   smftools.machine_learning.data.materialized_dataset
   smftools.machine_learning.data.partition_dataset
   smftools.machine_learning.data.preprocessing
   smftools.machine_learning.data.streaming_transforms
   smftools.machine_learning.data.transforms
```

## Model families and registry

Registered architectures, recipes, and artifact persistence.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.models.base
   smftools.machine_learning.models.cnn
   smftools.machine_learning.models.mlp
   smftools.machine_learning.models.positional
   smftools.machine_learning.models.protocols
   smftools.machine_learning.models.registry
   smftools.machine_learning.models.residual_cnn
   smftools.machine_learning.models.rnn
   smftools.machine_learning.models.sklearn_artifacts
   smftools.machine_learning.models.sklearn_models
   smftools.machine_learning.models.torch_artifacts
   smftools.machine_learning.models.transformer
   smftools.machine_learning.models.wrappers
```

## Training engines

sklearn and plain-PyTorch fits, materialized and streaming.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.training.sklearn_backend
   smftools.machine_learning.training.torch_backend
```

## Inference

Backend adapters and prediction records.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.inference.inference_utils
   smftools.machine_learning.inference.lightning_inference
   smftools.machine_learning.inference.sklearn_backend
   smftools.machine_learning.inference.sklearn_inference
   smftools.machine_learning.inference.torch_backend
```

## Evaluation

Predictor-neutral metrics, curves, folds, and histories.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.evaluation.contracts
   smftools.machine_learning.evaluation.eval_utils
   smftools.machine_learning.evaluation.evaluators
   smftools.machine_learning.evaluation.history
   smftools.machine_learning.evaluation.metrics
```

## Interpretability

Attribution requests, results, and backend adapters.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.interpretability.artifacts
   smftools.machine_learning.interpretability.background
   smftools.machine_learning.interpretability.classical
   smftools.machine_learning.interpretability.contracts
   smftools.machine_learning.interpretability.neural
```

## Artifacts

Immutable run, model, and result publication.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.artifacts.common
   smftools.machine_learning.artifacts.indexing
   smftools.machine_learning.artifacts.model
   smftools.machine_learning.artifacts.publication
   smftools.machine_learning.artifacts.results
   smftools.machine_learning.artifacts.run
```

## Orchestration

Backend-neutral job services, planning, and dispatch.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.orchestration.actions
   smftools.machine_learning.orchestration.contracts
   smftools.machine_learning.orchestration.planning
   smftools.machine_learning.orchestration.resolution
   smftools.machine_learning.orchestration.service
```

## Compatibility

Temporary adapters for the deprecated legacy surface.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.compatibility.classical_explanations
   smftools.machine_learning.compatibility.classical_models
   smftools.machine_learning.compatibility.matrix_cnn
```

## Benchmarks

Operator-invoked scale qualification. Not imported by pipeline code.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.benchmarks.fixtures
   smftools.machine_learning.benchmarks.harness
   smftools.machine_learning.benchmarks.sweeps
```

## Utilities

Device resolution and small shared helpers.

```{eval-rst}
.. autosummary::
   :toctree: generated/machine_learning
   :recursive:

   smftools.machine_learning.utils.device
   smftools.machine_learning.utils.grl
```

## Not listed here

Five modules are excluded because they cannot be imported under the documentation
build's mocked optional dependencies, and all five are surfaces on their way out or
not yet in:

| Module | Why |
| --- | --- |
| `data.anndata_data_module` | Legacy AnnData/Lightning data module; deprecated with a 3.0 removal window |
| `inference.sliding_window_inference` | Legacy inference helper; deprecated |
| `models.lightning_base` | PyTorch Lightning wrapper; Lightning is a deferred integration |
| `training.train_lightning_model` | Legacy Lightning trainer; deprecated |
| `training.train_sklearn_model` | Legacy sklearn trainer; deprecated |

Each subclasses a base from a package that the docs build mocks, and subclassing a
mock raises under Python 3.12. Since every one is either deprecated or gated behind
an integration that is not built, documenting them would advertise surfaces callers
should not adopt. The replacements are in the
[migration guide](../tutorials/ml_migration.md).
