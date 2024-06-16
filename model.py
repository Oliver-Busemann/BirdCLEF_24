import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
#from kaggle_metric_utilities import *
#from Birdclef_ROC_AUC import *
from torchmetrics.classification import MulticlassAUROC
from torch.optim.lr_scheduler import OneCycleLR



class Model(pl.LightningModule):

    def __init__(self, lr=4e-5, model_name='seresnext26d_32x4d'):
        super().__init__()

        self.learning_rate = lr

        self.model_name = model_name

        self.cnn = timm.create_model(self.model_name, pretrained=True, num_classes=182, in_chans=1)  # drop_rate=0.2

        # this metric is equal to the one in the competition when all classes exists
        self.metric = MulticlassAUROC(num_classes=182, average='macro')

        self.train_preds = []
        self.train_targets = []
        self.train_losses = []

        self.valid_preds = []
        self.valid_targets = []
        self.valid_losses = []

        # to calculate final CV-score for all folds
        self.last_valid_macro_AUROC = torch.zeros((1))


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
        preds = torch.cat(self.train_preds, dim=0)
        targets = torch.cat(self.train_targets, dim=0)

        macro_AUROC = self.metric(preds, targets)

        loss = torch.mean(torch.as_tensor(self.train_losses))

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


    def on_validation_epoch_end(self):
        
        preds = torch.cat(self.valid_preds, dim=0)
        targets = torch.cat(self.valid_targets, dim=0)

        # the targets need to be onehot vectors instead of the class numbers
        #targets = F.one_hot(targets, num_classes=182)

        macro_AUROC = self.metric(preds, targets)

        self.last_valid_macro_AUROC = macro_AUROC

        loss = torch.mean(torch.as_tensor(self.valid_losses))

        self.log_dict({'Valid_loss': loss, 'Valid_macro_AUROC': macro_AUROC}, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.valid_preds = []
        self.valid_targets = []
        self.valid_losses = []

    def configure_optimizers(self):
        
        optim = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)
        
        lr_scheduler = {
            'scheduler': OneCycleLR(
                optimizer=optim,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=1  # self.trainer.estimated_stepping_batches
                ),
            'name': 'Learning Rate'
            }

        return [optim], [lr_scheduler]
