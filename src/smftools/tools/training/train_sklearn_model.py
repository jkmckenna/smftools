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
    # Prepare data
    datamodule.setup()
    
    # Extract train data
    train_set = datamodule.train_set
    X_tensor, y_tensor = train_set.dataset.X_tensor, train_set.dataset.y_tensor
    indices = train_set.indices
    X_train = X_tensor[indices].numpy()
    y_train = y_tensor[indices].numpy()

    # Fit model
    model_wrapper.fit(X_train, y_train)

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
