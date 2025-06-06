import torch
import pytorch_lightning as pl

class TorchClassifierWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=None,
        criterion_cls=torch.nn.CrossEntropyLoss,
        criterion_kwargs=None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])  # logs all except actual model instance
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.criterion = criterion_cls(**(criterion_kwargs or {}))
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
