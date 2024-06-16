import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np


folder_data = '/home/olli/Projects/Kaggle/BirdCLEF/Data'

csv_name = 'Processed_5s_Spectrograms_Split_small.csv'  # 'Processed_5s_Spectrograms_Split_small.csv' 'Processed_5s_Spectrograms_Split_RemovedZerosMeldB.csv'

DF = pd.read_csv(os.path.join(folder_data, csv_name))

# create the targets with a dictionary, i.e. numbers instead of strings
targets = {word: num for word, num in zip(sorted(list(DF.target.unique())), range(len(DF.target.unique())))}

# augmentations for training
train_aug = A.Compose([
    A.XYMasking(
        p=0.5,
        num_masks_x=(1, 2),
        num_masks_y=(1, 2),
        mask_x_length=(3, 15),
        mask_y_length=(3, 15),
    )
])


class Data(Dataset):

    def __init__(self, DF, data_path, transform=None, targets=targets):

        self.df = DF

        # shuffle the df and reset indices for getitem index
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df = self.df.reset_index(drop=True)

        self.transform = transform
        self.data_path = data_path  # column of the data to use (spec, spec_db, mel, mel_db)
        self.targets = targets

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        # load the spectrogram of choice
        path = self.df.at[index, self.data_path]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # augment it
        if self.transform:
            img = self.transform(image=img)['image']

        # normalize it; epsilon needed when using mel specs with masking augmentation
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # get the target number
        y = self.targets[self.df.at[index, 'target']]

        # get correct shape and type
        img = torch.Tensor(img).unsqueeze(0).type(torch.float32)

        return img, y
    

class LightningData(pl.LightningDataModule):

    def __init__(self, ds_train, ds_valid, batch_size=32, num_workers=32, pin_memory=True):

        super().__init__()
        self.ds_train = ds_train
        self.ds_valid = ds_valid

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):

        # assign the sampler for train to upsample minority classes
        dl_train = DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=WeightedRandomSampler(
                weights=self.ds_train.df.weights.values,
                num_samples=self.ds_train.__len__(),
                replacement=True  # with false samples can only be sampled once which is wrong
                )
        )

        return dl_train

    def val_dataloader(self):

        dl_valid = DataLoader(
            dataset=self.ds_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        return dl_valid


# show some augmantations when running script
if __name__ == '__main__':

    ds = Data(DF=DF, data_path='path_spec_db', transform=train_aug)

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 5))

    ax = ax.flatten()

    for i in range(10):

        img, _ = ds.__getitem__(i)

        img = (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)

        ax[i].imshow(img, cmap='jet')

    plt.show()