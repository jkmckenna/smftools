import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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