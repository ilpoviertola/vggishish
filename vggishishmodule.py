import pytorch_lightning as pl
import torch

from vggishish import VGGishish


class VGGishishModule(pl.LightningModule):
    def __init__(
        self,
        conv_layers: list,
        use_bn: bool,
        num_classes: int,
        lr: float,
        betas: list,
        weight_decay: float,
        batch_size: int,
    ):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.model = VGGishish(conv_layers, use_bn, num_classes)

        self.test_total = 0
        self.test_correct = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["spec"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["spec"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_total = 0
        self.test_correct = 0

    def test_step(self, batch, batch_idx):
        x = batch["spec"]
        y = batch["label"]
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        _, predicted = torch.max(y_hat.data, 1)

        self.test_total += y.size(0)
        self.test_correct += (predicted == y).sum().item()

        return loss

    def on_test_epoch_end(self) -> None:
        self.log(
            "test_acc",
            self.test_correct / self.test_total,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
