import torch
import pytorch_lightning as pl
from data import *
from model import *
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import numpy as np


NAME = 'Baseline'

BATCH_SIZE = 32
NUM_WORKERS = 32
PIN_MEMORY = True
LEARNING_RATE = 5e-4
EPOCHS = 1


df = DF.copy()

scores = []

# loop over all folds, and create train/val-split
for fold in range(1, 6):
    df_train = df[df['fold'] != fold]  # all 4 other folds are train
    df_valid = df[df['fold'] == fold]

    ds_train = Data(DF=df_train, data_path='path_mel_db', transform=None, targets=targets)
    ds_valid = Data(DF=df_valid, data_path='path_mel_db', transform=None, targets=targets)

    data_module = LightningData(
                            ds_train=ds_train, 
                            ds_valid=ds_valid,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY
                            )
    
    model = Model(lr=LEARNING_RATE)
    
    logger = TensorBoardLogger(
                        save_dir='/home/olli/Projects/Kaggle/BirdCLEF/tb_logs',
                        name=NAME,
                        version=f'Fold_{fold}'
                        )

    trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=[0],
                    precision='16-mixed',
                    logger=logger,
                    max_epochs=EPOCHS,
    )

    trainer.fit(model=model, datamodule=data_module)

    scores.append(trainer.model.last_valid_macro_AUROC)

print(scores)
print(np.array(scores).mean())