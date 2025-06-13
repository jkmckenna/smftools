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
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

    if evaluate_test:
        test_metrics = model_wrapper.evaluate_from_datamodule(datamodule, split="test")
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Plot evaluations
    model_wrapper.plot_roc_pr_curves()

    return metrics
