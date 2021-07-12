import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from PIL import Image
import numpy as np

# Define transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class ResNet(pl.LightningModule):
    def __init__(self,
                 pretrained: bool = False,
                 num_classes: int = 5,
                 lr: float = 2e-4,
                 l2_norm: float = 0.0,
                 lr_scheduler_factor: float = 0.2,
                 lr_scheduler_patience: int = 8,
                 lr_scheduler_min_lr: float = 1e-11
                 ):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.hparams.lr = lr
        self.hparams.l2_norm = l2_norm
        self.hparams.lr_scheduler_factor = lr_scheduler_factor
        self.hparams.lr_scheduler_patience = lr_scheduler_patience
        self.hparams.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.save_hyperparameters()
        self.ce = torch.nn.CrossEntropyLoss()

        self.x = None

        # Define model
        self.model = models.resnet50(pretrained=False)
        # Replace last layer with one that has the required number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        # Define transforms
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Utility function for creating a tensor from a list of image paths
    @staticmethod
    def img_etl(path_to_image):
        img = Image.open(path_to_image)
        img = transform(img)
        return img

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        preds = F.softmax(preds, dim=1)
        loss = self.ce(preds, y)
        self.log('train_loss', loss)

        # Calculate and log accuracy
        self.log('train_acc', pl.metrics.functional.accuracy(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        preds = F.softmax(preds, dim=1)
        loss = self.ce(preds, y)
        self.log('validation_loss', loss)

        # Calculate and log accuracy
        self.log('validation_acc', pl.metrics.functional.accuracy(preds, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        preds = F.softmax(preds, dim=1)
        loss = self.ce(preds, y)
        self.log('test_loss', loss)

        # Calculate and log accuracy
        self.log('test_acc', pl.metrics.functional.accuracy(preds, y))
        return loss

    def forward(self,
                image_paths: list):
        x = [self.img_etl(path) for path in image_paths]
        x = torch.stack(x)
        preds = self.model(x)
        class_probs = np.array(F.softmax(preds, dim=1))
        class_preds = np.array(np.argmax(preds, axis=1))
        out = {'class_probabilities': class_probs,
               'class_predictions': class_preds}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.l2_norm)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=self.hparams.lr_scheduler_factor,
                                                                  patience=self.hparams.lr_scheduler_patience,
                                                                  min_lr=self.hparams.lr_scheduler_min_lr,
                                                                  verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'validation_loss',
            'reduce_on_plateau': True
        }

        return [optimizer], [scheduler]

