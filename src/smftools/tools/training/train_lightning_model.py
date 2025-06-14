import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ..data import AnnDataModule
from ..models import TorchClassifierWrapper

def train_lightning_model(
    model,
    datamodule,
    max_epochs=30,
    patience=5,
    monitor_metric="val_loss",
    checkpoint_path=None,
    evaluate_test=True,
    devices=1
):
    """
    Takes a PyTorch Lightning Model and a Lightning DataLoader module to define a Lightning Trainer.
    - The Lightning trainer fits the model to the training split of the datamodule.
    - The Lightning trainer uses the validation split of the datamodule for monitoring training loss.
    - Option of evaluating the trained model on a test set when evaluate_test is True.
    - When using cuda, devices parameter can be: 1, [0,1], "all", "auto". Depending on what devices you want to use.
    """
    # Device logic
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    # adds the train/val/test indices from the datamodule to the model class.
    model.set_training_indices(datamodule)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=patience, mode="min"),
    ]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="{epoch}-{val_loss:.4f}",
            monitor=monitor_metric,
            save_top_k=1,
            mode="min",
        ))

    # Trainer setup
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        enable_progress_bar=False
    )

    # Fit model with trainer
    trainer.fit(model, datamodule=datamodule)

    # Test model (if applicable)
    if evaluate_test and hasattr(datamodule, "test_dataloader"):
        trainer.test(model, datamodule=datamodule)
    
    # Return best checkpoint path
    best_ckpt = None
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_ckpt = cb.best_model_path

    return trainer, best_ckpt

def run_sliding_window_lightning_training(
    adata,
    tensor_source,
    tensor_key,
    label_col,
    model_class,
    num_classes,
    class_names,
    class_weights,
    focus_class,
    window_size,
    stride,
    max_epochs=30,
    patience=5,
    enforce_eval_balance: bool=False,
    target_eval_freq: float=0.3,
    max_eval_positive: int=None
):
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
            batch_size=64,
            window_start=start,
            window_size=window_size
        )
        datamodule.setup()

        # Build model for this window
        model = model_class(window_size, num_classes)
        wrapper = TorchClassifierWrapper(
            model, label_col=label_col, num_classes=num_classes,
            class_names=class_names,
            class_weights=class_weights,
            focus_class=focus_class, enforce_eval_balance=enforce_eval_balance,
            target_eval_freq=target_eval_freq, max_eval_positive=max_eval_positive
        )

        # Train model
        trainer, ckpt = train_lightning_model(
            wrapper, datamodule, max_epochs=max_epochs, patience=patience
        )

        results[center_varname] = {
            "model": wrapper,
            "trainer": trainer,
            "checkpoint": ckpt,
            "metrics": trainer.callback_metrics
        }
    
    return results
