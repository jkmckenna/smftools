from ..data import AnnDataModule
from ..models import SklearnModelWrapper

def train_sklearn_model(
    model_wrapper, 
    datamodule, 
    evaluate_test=True, 
    evaluate_val=False
):
    """
    Fits a SklearnModelWrapper on the train split from datamodule.
    Evaluates on test and/or val set.
    
    Parameters:
        model_wrapper: SklearnModelWrapper instance
        datamodule: AnnDataModule instance (with setup() method)
        evaluate_test: whether to evaluate on test split
        evaluate_val: whether to evaluate on validation split

    Returns:
        metrics: dictionary containing evaluation metrics
    """
    # Fit model
    model_wrapper.fit_from_datamodule(datamodule)

    # Evaluate
    metrics = {}

    if evaluate_val:
        val_metrics = model_wrapper.evaluate_from_datamodule(datamodule, split="val")
        metrics.update({f"{k}": v for k, v in val_metrics.items()})

    if evaluate_test:
        test_metrics = model_wrapper.evaluate_from_datamodule(datamodule, split="test")
        metrics.update({f"{k}": v for k, v in test_metrics.items()})

    # Plot evaluations
    model_wrapper.plot_roc_pr_curves()

    return metrics

def run_sliding_window_sklearn_training(
    adata,
    tensor_source,
    tensor_key,
    label_col,
    model_class,
    num_classes,
    class_names,
    focus_class,
    window_size,
    stride,
    batch_size=64,
    train_frac=0.6,
    val_frac=0.1,
    test_frac=0.3,
    random_seed=42,
    enforce_eval_balance=False,
    target_eval_freq=0.3,
    max_eval_positive=None,
    **model_kwargs
):
    """
    Sliding window training for sklearn models using AnnData.

    Returns dict keyed by window center.
    """

    input_len = adata.shape[1]
    results = {}

    for start in range(0, input_len - window_size + 1, stride):
        center_idx = start + window_size // 2
        center_varname = adata.var_names[center_idx]
        print(f"\nTraining window around {center_varname}")

        # Build datamodule for this window
        datamodule = AnnDataModule(
            adata,
            tensor_source=tensor_source,
            tensor_key=tensor_key,
            label_col=label_col,
            batch_size=batch_size,
            window_start=start,
            window_size=window_size,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            random_seed=random_seed
        )
        datamodule.setup()

        # Build model wrapper
        sklearn_model = model_class(**model_kwargs)
        wrapper = SklearnModelWrapper(
            sklearn_model, 
            num_classes=num_classes,
            label_col=label_col,
            class_names=class_names,
            focus_class=focus_class,
            enforce_eval_balance=enforce_eval_balance,
            target_eval_freq=target_eval_freq,
            max_eval_positive=max_eval_positive
        )

        # Fit and evaluate
        metrics = train_sklearn_model(wrapper, datamodule, evaluate_test=True, evaluate_val=False)

        results[center_varname] = {
            "model": wrapper,
            "metrics": metrics
        }

    return results
