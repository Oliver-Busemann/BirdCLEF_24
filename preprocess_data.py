'''
In this script all raw audio files will be processed and saved as a specrogram and mel spectrogram for later use.
Additionally dB-scaled ones will be created.
A csv will be created that contains all the necessary information. 
'''

import librosa
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from playsound import playsound
import math
import numpy as np
import cv2
from PIL import Image

N_MELS = 128
IMG_SIZE = 256

folder = '/home/olli/Projects/Kaggle/BirdCLEF'
folder_data = os.path.join(folder, 'Data')

# folders to save the preprocessed spectrograms to
folder_spec = os.path.join(folder_data, 'Spectrograms')
folder_spec_db = os.path.join(folder_data, 'Spectrograms_dB')
folder_mel = os.path.join(folder_data, 'Mel_Spectrogram')
folder_mel_db = os.path.join(folder_data, 'Mel_Spectrogram_dB')

# create the folders
for f in [folder_spec, folder_spec_db, folder_mel, folder_mel_db]:
    os.makedirs(f, exist_ok=True)

# read the metadata
path_meta = os.path.join(folder, 'train_metadata.csv')
df = pd.read_csv(path_meta)

# first remove some dublicates (19 of 24459 rows have dublicate ogg files - these are in different folders tho i.e. classes)
df['file'] = df['filename'].apply(lambda x: x.split('/')[1].split('.')[0])  # filenames to find dublicates
df = df.drop_duplicates(subset=['file'])  # remove them

# full path to the raw ogg files
df['path_ogg'] = df['filename'].apply(lambda x: os.path.join(folder, 'train_audio', x))

# columns of the df with the final training data
cols = ['path_spec', 'path_spec_db', 'path_mel', 'path_mel_db', 'file', 'img_num', 'rating', 'sr', 'target']
df_save = pd.DataFrame({col: [] for col in cols})

# "files have been downsampled to 32 kHz where applicable" -> if not do so
hz = 32_000

# process the whole data
for index in tqdm(df.index):

    rating = df.at[index, 'rating']
    file = df.at[index, 'file']
    target = df.at[index, 'primary_label']

    # load the ogg file to process it
    y, sr = librosa.load(df.at[index, 'path_ogg'], sr=None, mono=True)
    
    # sample rate should be 36k Hz; if not change it
    if hz != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=hz)
        print(f'File {file} had a sr of {sr} Hz; resampling to {hz} Hz.')

    # append 5s sequences here to process them all
    sequences = []

    # smaller then 1s so skip this one
    if len(y) < hz:
        continue

    # now if the file is shorter then 5s, we dublicate it and then crop it to exactly 5s
    elif len(y) < 5 * hz:
        missing = 5 * hz - len(y)
        add = math.ceil(missing / len(y)) + 1  # if we need to add it e.g. 3.4 times, we add it 4 times and slice off the rest
        y = np.tile(y, add)
        y = y[:5 * hz]
        sequences.append(y)
    
    # if the audeio is longer then 5s, we append all every 5s chunk
    else:
        for i in range(0, len(y), 5 * hz):
            sequences.append(y[i: i + 5 * hz])

        # the last part is smaller then 5*hz; so if its long enough include it as well from slicing backwards (>1s)
        if len(y) % (5 * hz) > hz:
            sequences.append(y[-5 * hz:])

    # now loop over all 5s slices and process them
    for num, seq in enumerate(sequences):

        # create the spectrograms
        spec = np.abs(librosa.stft(y=seq))
        spec_db = librosa.amplitude_to_db(spec)
        mel = librosa.feature.melspectrogram(y=seq, sr=hz, n_mels=N_MELS)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)

        # normalize them to values between 0 and 1
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
        spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db))
        mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
        mel_db = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db))

        # create images from that
        spec = (spec * 255).astype(np.uint8)
        spec_db = (spec_db * 255).astype(np.uint8)
        mel = (mel * 255).astype(np.uint8)
        mel_db = (mel_db * 255).astype(np.uint8)

        # resize them to the correct shape
        spec = cv2.resize(spec, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        spec_db = cv2.resize(spec_db, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mel = cv2.resize(mel, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mel_db = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # create PIL Image to save them
        spec = Image.fromarray(spec)
        spec_db = Image.fromarray(spec_db)
        mel = Image.fromarray(mel)
        mel_db = Image.fromarray(mel_db)

        # create filepaths to save them
        spec_path = os.path.join(folder_spec, f'spec_{file}_{num}.png')
        spec_db_path = os.path.join(folder_spec_db, f'spec_dB_{file}_{num}.png')
        mel_path = os.path.join(folder_mel, f'mel_{file}_{num}.png')
        mel_db_path = os.path.join(folder_mel_db, f'mel_db_{file}_{num}.png')

        # save them
        spec.save(spec_path)
        spec_db.save(spec_db_path)
        mel.save(mel_path)
        mel_db.save(mel_db_path)

        # now add it as an entry to the df_save
        values_save = [spec_path, spec_db_path, mel_path, mel_db_path, file, num, rating, sr, target]
        
        df_tmp = pd.DataFrame({c: [v] for c, v in zip(cols, values_save)})
        df_save = pd.concat([df_save, df_tmp], axis=0)

# after all files are saved and the entries are added to df_save we save it as a csv
df_save.reset_index(drop=True, inplace=True)

df_save.to_csv(os.path.join(folder_data, 'Processed_5s_Spectrograms.csv'), index=False)
