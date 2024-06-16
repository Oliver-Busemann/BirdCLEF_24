# BirdCLEF_24

https://www.kaggle.com/competitions/birdclef-2024

### 1) preprocess_data.py
-From the raw audio data create spectrograms and save them to disk</br>
-This is done for each 5s sequence</br>
-The paths to the spectrograms/mel-spectrograms and additional information like the target and rating of the recording are saved as a csv</br>

### 2) split_data.py
-The preprocessed data must now be split into 5-folds</br>
-Each fold should rougly have the same class distribution</br>
-Additionally multiple files from the same recording must be a in the same fold to avoid leakage</br>
-If specified only the first and last 5s crop from the raw audio will be used to reduce the size</br>

### 3) data.py
-Create a lightning dataset to define the data loading process</br>
-The dataset takes as input a df where each row is one datapoint</br>
-Spectrogram augmentation (only x & y masking worked)</br>

### 4) model.py
-model to train is build using timm (1channel input; 182 outputs)</br>
-lightning module is created that defines the training and validation steps</br>
-metric to use is MulticlassAUROC with average macro; this doesnt differ from the competition metric since no class is missing in training/validation</br>
-LR-scheduler (OneCycleLR)</br>
-All metrics will be logged to tensorboard after each epoch</br>

### 5) train.py
-train a 5-fold model: simply loop over each fold specified in the df and create a train/valid-df out of it</br>
-upsample minority classes with the WeightedRandomSampler from torch</br>
-upsample files the have a high rating (i.e. quality)</br>
-when using all files (not just the 1st and last from the raw audio) upsample the short audio sequences</br>
-finally print the cv-score after all 5 models are evaluated</br>
-if specified retrain on the full data and save the model weights (5 models are not feasible for inference)</br>

### 6) export_weights.py
-for faster inference quantize the model weights with onnx and save them</br>
