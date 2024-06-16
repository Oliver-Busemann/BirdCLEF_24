# BirdCLEF_24

https://www.kaggle.com/competitions/birdclef-2024

### 1) preprocess_data
-From the raw audio data create spectrograms and save them to disk.
-This is done for each 5s sequence.
-The paths to the spectrograms/mel-spectrograms and additional information like the target and rating of the recording are saved as a csv.

### 2) split_data
-The preprocessed data must now be split into 5-folds
-Each fold should rougly have the same class distribution
-Additionally multiple files from the same recording must be a in the same fold to avoid leakage
-If specified only the first and last 5s crop from the raw audio will be used to reduce the size

### 3) 
