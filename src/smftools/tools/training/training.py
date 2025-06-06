import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def train_lightning_model(
    model,
    datamodule,
    max_epochs=20,
    patience=5,
    monitor_metric="val_loss",
    checkpoint_path=None,
):
    # Device logic
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

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
    )
    trainer.fit(model, datamodule=datamodule)
    
    return trainer
