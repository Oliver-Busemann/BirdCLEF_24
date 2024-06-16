import torch
import pytorch_lightning as pl
from data import *
from model import *
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import numpy as np
import gc
from pytorch_lightning.callbacks import LearningRateMonitor


NAME = 'MelDB_EffNetB0_Train_Full'  # 'Baseline_2_XYMask_UpsampleHighRating'

TRAIN_FULL = True  # train with full data and save weights for submission

BATCH_SIZE = 32
NUM_WORKERS = 32
PIN_MEMORY = True
LEARNING_RATE = 1.7e-5
EPOCHS = 10
TYPE_DATA = 'path_spec_db'  # path to the files (path_spec, path_spec_db, path_mel, path_mel_db)
MODEL_NAME = 'efficientnet_b0'  # seresnext26d_32x4d

if TRAIN_FULL:
    LIMIT_VAL_BATCHES = 0  # dont use validation loop
else:
    LIMIT_VAL_BATCHES = None  # use validation loop

torch.set_float32_matmul_precision('medium')

df = DF.copy()

scores = []

# loop over all folds, and create train/val-split
for fold in range(1, 6):
    df_train = df[df['fold'] != fold].copy()  # all 4 other folds are train
    df_valid = df[df['fold'] == fold].copy()

    if TRAIN_FULL:
        # use all data
        df_train = df.copy()

    # calculate the class counts to upsample the minority ones and create weights for them
    class_weights = {c: 1 / w for c, w in zip(df_train.target.value_counts().index, df_train.target.value_counts().values)}
    
    # the weightedrandomsampler needs for each sample one weight in a list (for all samples)
    weights = [class_weights[target] for target in df_train.target.values]

    # since all rows will be shuffle for additional randomness we need to assign the weights as a col so they are shuffled the same way
    df_train['weights'] = weights
    
    # now try to upsample the files in each class that have a higher rating (i.e. better quality)
    # for this we first assign the ratings with 0 (unknown) a 1 (bad quality)
    df_train['rating'] = df_train['rating'].apply(lambda x: 1 if x == 0 else x)
    df_train['rating'] = df_train['rating'].apply(lambda x: x + 0.5)  # now ratings are from 1 to 5.5

    # adjust the weights so that ratings of 5 are sampled 5-times more often then a rating of 1
    df_train['weights'] = df_train['weights'] * df_train['rating']

    # now we need to adjust the weights again so that all samples of each class sums up to 1
    df_train['weights'] = df_train.groupby('target')['weights'].transform(lambda x: x / x.sum())

    # when we use all the files from the raw audio, long sequences are overrepresented
    # so upsample the shorter files, so that they are sampled more often (min is 1 max is 1194; average number of 5s audios per file is 142)
    df_train['weights'] = df_train['weights'] * (1 / df['file_count'])
    df_train['weights'] = df_train.groupby('target')['weights'].transform(lambda x: x / x.sum())

    ds_train = Data(DF=df_train, data_path=TYPE_DATA, transform=train_aug, targets=targets)
    ds_valid = Data(DF=df_valid, data_path=TYPE_DATA, transform=None, targets=targets)

    data_module = LightningData(
        ds_train=ds_train, 
        ds_valid=ds_valid,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
        )
    
    model = Model(
        lr=LEARNING_RATE,
        model_name=MODEL_NAME,
        )
    
    logger = TensorBoardLogger(
        save_dir='/home/olli/Projects/Kaggle/BirdCLEF/tb_logs',
        name=NAME,
        version=f'Fold_{fold}'
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        precision='16-mixed',
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[lr_monitor],
        limit_val_batches=LIMIT_VAL_BATCHES
    )

    trainer.fit(model=model, datamodule=data_module)

    # append final macro AUROC to calculate CV
    scores.append(float(model.last_valid_macro_AUROC.cpu().numpy()))

    gc.collect()
    torch.cuda.empty_cache()

    # save weights
    if TRAIN_FULL:
        save_path = f'/home/olli/Projects/Kaggle/BirdCLEF/Model_Weights/{NAME}.pth'

        # inference in kaggle is done on cpu
        model.cnn.to('cpu')
        torch.save(model.cnn.state_dict(), save_path)

        break

print(np.array(scores).mean())
