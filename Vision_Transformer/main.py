from pl_bolts.datamodules import CIFAR10DataModule
from torch.optim.lr_scheduler import OneCycleLR
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from ViT import ViT
from torchmetrics.functional import accuracy
import torch
import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')  # option


config = {"data_dir": ".", "batch_size": 256, "num_workers": 2, "num_classes": 10, "lr": 1e-4, "max_lr": 1e-3}

train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                              cifar10_normalization()])  # perturbation

test_transforms = T.Compose([T.ToTensor(), cifar10_normalization()])

# Train: 45,000, valid: 5,000, test: 10,000
cifar10_dm = CIFAR10DataModule(
    data_dir=config["data_dir"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms
)


# Defines a LightningModule for training a ViT model
class LitViT(pl.LightningModule):
    # Initialize model and hyperparameters
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViT(in_channels=3, patch_size=4, emb_size=128, img_size=32, depth=12, num_classes=10)

    # Forward pass
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # model forward pass
        loss = F.nll_loss(logits, y)  # negative log likelihood loss
        self.log("train_loss", loss)
        return loss

    # Shared evaluation logic for validation and test
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=config["num_classes"])

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    # Validation step
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    # Test step
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # Configure optimizer and learning rate scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        steps_per_epoch = 45_000 // config["batch_size"]
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=config["max_lr"],
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":

    model = LitViT(lr=config["lr"])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="vit-best",
        dirpath="checkpoints"
    )
    trainer = pl.Trainer(max_epochs=30, accelerator="auto", callbacks=[checkpoint_callback])

    # Auto LR finder
    lr_finder = trainer.tuner.lr_find(model, cifar10_dm)
    model.hparams.lr = lr_finder.suggestion()# Auto-find model LR is: 0.000630957344480193
    trainer.fit(model, cifar10_dm)

    ckpt_path = "checkpoints/vit-best.ckpt"
    model = LitViT.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(accelerator="auto")
    trainer.test(model, datamodule=cifar10_dm)






