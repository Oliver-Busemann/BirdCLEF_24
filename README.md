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

### 3) data
-Create a lightning dataset to define the data loading process
-The dataset takes as input a df where each row is one datapoint
-Spectrogram augmentation (only x & y masking worked)

### 4) model
-model to train is build using timm (1channel input; 182 outputs)
-lightning module is created that defines the training and validation steps
-metric to use is MulticlassAUROC with average macro; this doesnt differ from the competition metric since no class is missing in training/validation
-LR-scheduler (OneCycleLR)
-All metrics will be logged to tensorboard after each epoch

### 5) train

