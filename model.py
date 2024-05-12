import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from kaggle_metric_utilities import *
from Birdclef_ROC_AUC import *
from torchmetrics.classification import MulticlassAUROC



class Model(pl.LightningModule):

    def __init__(self, lr=4e-5):
        super().__init__()

        self.learning_rate = lr

        self.cnn = timm.create_model('seresnext26d_32x4d', pretrained=True, num_classes=182, drop_rate=0.2, in_chans=1)

        # this metric is equal to the one in the competition when all classes exists
        self.metric = MulticlassAUROC(num_classes=182, average='macro')

        self.train_preds = []
        self.train_targets = []
        self.train_losses = []

        self.valid_preds = []
        self.valid_targets = []
        self.valid_losses = []


    def forward(self, x):
        
        return self.cnn(x)
    
    def training_step(self, batch, batch_idx):

        X, y = batch

        pred = self.cnn(X)

        loss = F.cross_entropy(pred, y)

        pred_sm = torch.softmax(pred, dim=1)

        self.train_preds.append(pred_sm)
        self.train_targets.append(y)
        self.train_losses.append(loss)

        return loss

    def on_train_epoch_end(self):

        # calculate the metric for all samples together
        preds = torch.stack(self.train_preds, dim=0)
        targets = torch.stack(self.train_targets, dim=0)

        macro_AUROC = self.metric(preds, targets)

        loss = torch.mean(torch.stack(self.train_losses))

        self.log_dict({'Train_loss': loss, 'Train_macro_AUROC': macro_AUROC}, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.train_preds = []
        self.train_targets = []
        self.train_losses = []


    def validation_step(self, batch, batch_idx):

        X, y = batch

        pred = self.cnn(X)

        loss = F.cross_entropy(pred, y)

        pred_sm = torch.softmax(pred, dim=1)

        self.valid_preds.append(pred_sm)
        self.valid_targets.append(y)
        self.valid_losses.append(loss)

    def on_valid_epoch_end(self):

        preds = torch.stack(self.valid_preds, dim=0)
        targets = torch.stack(self.valid_targets, dim=0)

        macro_AUROC = self.metric(preds, targets)

        loss = torch.mean(torch.stack(self.train_losses))

        self.log_dict({'Valid_loss': loss, 'Valid_macro_AUROC': macro_AUROC}, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.valid_preds = []
        self.valid_targets = []
        self.valid_losses = []

    def configure_optimizers(self):
        
        optim = torch.optim.Adam(lr=self.learning_rate)

        return optim
