# Choosing an interpretability method

Attribution methods answer different questions, and most disagreement between them is the methods
answering different questions rather than one being wrong. This page is about picking the one that
matches your question.

## Start from the question

| Your question | Method | Backend |
| --- | --- | --- |
| Which features does this linear or NB model weight, globally? | `NaiveBayesLogOdds`, `LinearCoefficients` | sklearn |
| How much does each feature actually contribute to held-out performance? | `PermutationImportance` | sklearn |
| How did this tree ensemble reach this prediction? | `TreeSHAP` | sklearn |
| Which positions drove *this read's* prediction? | `Saliency`, `InputXGradient` | Torch |
| Same, but with a defensible attribution baseline | `IntegratedGradients`, `DeepLift`, `GradientSHAP` | Torch |
| Which regions does a convolutional layer respond to? | `LayerGradCam`, `GuidedGradCam` | Torch |

## Global structure versus per-read explanation

`NaiveBayesLogOdds` and `LinearCoefficients` read parameters out of a fitted model. They are exact,
free, and describe the model globally — they say nothing about an individual read.

`PermutationImportance` measures something different and often more useful: how much held-out
performance degrades when a feature is shuffled. It is computed on validation or test rows, never
train, so it reports what a feature is worth to generalisation rather than to memorisation. It
costs one evaluation pass per permuted feature.

The gradient methods are per-observation. They tell you which positions moved *this* prediction,
which is what you want when asking why one molecule was called active.

## Baselines are a scientific choice

`IntegratedGradients`, `DeepLift`, and `GradientSHAP` attribute relative to a **baseline** — an
input representing "absent". The attribution is a comparison, so the baseline is part of the
result's meaning, not a tuning parameter.

The package requires a checksummed **training background** for these methods and records its
identity in the result. A request that declares a baseline without supplying the background raises,
and so does the reverse. That is deliberate: an attribution whose baseline you cannot reconstruct
is not reproducible.

`Saliency` and `InputXGradient` need no baseline, which makes them cheaper and easier to interpret,
but they are local gradients — they describe sensitivity at the current input, not contribution
relative to anything.

## Masks are respected, not silently averaged

Attributions are zeroed at positions that are not observed, not available, outside the read's span,
or not design sites for the channel. An unobserved position gets no attribution rather than an
attribution of zero that reads as "no effect" — the distinction matters when a fifth of a read is
unmeasured.

## Chunk size changes results

`example_batch_size` controls how many observations are attributed at once. Two properties are easy
to get wrong:

- **Wall time has an interior optimum. Larger is not faster.** Measured over 400 rows × 400
  positions with Saliency, a chunk of 8 was fastest; 1 took 1.99× longer and 512 took 2.76× longer.
- **It changes attribution values.** Results repeat bitwise at a fixed chunk size but differ by
  roughly 1.1×10⁻³ relative across chunk sizes, consistent with batch-size-dependent kernel
  selection accumulating through the network.

Provenance stays honest — `example_batch_size` is part of the request and therefore its identity,
so two chunk sizes are recorded as different results rather than colliding. But **pick one and hold
it across runs you intend to compare**, or you are comparing attributions that differ for a reason
unrelated to your biology.

## Optional dependencies and capability gating

Captum backs the neural methods and SHAP backs `TreeSHAP`. Both are optional `ml-extended`
dependencies, imported lazily *after* the model, request, schema, masks, layer, parameters, and
background have passed preflight — so a missing extra fails with an actionable message rather than
an import error partway through a run.

Methods are also gated on model capability. `LayerGradCam` is available only for a model declaring
a compatible convolutional `attribution_layer`. Attention-based explanations are **not available**:
no registered model exposes validated attention, and raw attention weights are not emitted as
explanations, because attention is not by itself an explanation of a prediction.

## Reading the result

An `AttributionResult` carries values aligned to genomic positions and channels, the request that
produced it, and — where the method supports it — a convergence delta. Check the delta for
`IntegratedGradients`: a large value means the integral did not converge and the attribution should
not be trusted at face value.
