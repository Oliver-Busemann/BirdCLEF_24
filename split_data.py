'''
This script will load the created csv with all the training data and split it into 5 folds.
The split is done in a way that all crops of a recording are in a single fold to avoid leakage.
Futhermore the class distribution should be as equal as possible based on the above boundary condition. 
'''

import pandas as pd
import os
import numpy as np
from tqdm import tqdm


folder_data = '/home/olli/Projects/Kaggle/BirdCLEF/Data'

DF = pd.read_csv(os.path.join(folder_data, 'Processed_5s_Spectrograms.csv'))
df = DF.copy()

# now 5-folds need to be created; all files from one original recording should be in the same fold
# additionally the distribution of classes should roughly be the same
# to archieve this we create a custom loop and assign the files individually
df['file_count'] = df.groupby('file')['file'].transform('count')  # add the filecount for sorting
df['fold'] = None

for target in tqdm(df.target.unique()):
    df_tmp = df[df['target'] == target]

    # now sort it that the most common file is at the top, the second most common below that one and so on
    df_tmp = df_tmp.sort_values(by=['file_count', 'file'], ascending=[False, True])
    df_tmp.reset_index(drop=True, inplace=True)

    # we need a count for which fold to append all crops from a file
    count = [0, 0, 0, 0, 0]

    for file in df_tmp.file.unique():

        # index 0 is fold 1
        fold = np.argmin(count) + 1

        # get number of crops for this file
        num_files = df_tmp[df_tmp['file'] == file]['file_count'].values[0]

        # add the number of files to the count
        count[fold - 1] += num_files

        # now assign the fold to the original df
        df.loc[(df['target'] == target) & (df['file'] == file), 'fold'] = fold

df.to_csv('/home/olli/Projects/Kaggle/BirdCLEF/Data/Processed_5s_Spectrograms_Split.csv', index=False)