## This script is meant to run a preliminary convolutional neural network model on 
## the training data for the diabetic retinopathy detection competition for APTOS.

import numpy as np 
import pandas as pd

import os

import tensorflow as tf

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile, rmtree



train = pd.read_csv('../input/train.csv')

os.mkdir('../data')
os.mkdir('../data/train')
os.mkdir('../data/validation')
os.mkdir('../data/train/positives')
os.mkdir('../data/train/negatives')
os.mkdir('../data/validation/positives')
os.mkdir('../data/validation/negatives')

positives = train[train['diagnosis'] != 0]
negatives = train[train['diagnosis'] == 0]

positive_filenames = list(positives.id_code + '.png')
negative_filenames = list(negatives.id_code + '.png')

positive_filenames_train = np.random.choice(positive_filenames, int(len(positive_filenames)*0.8))
positive_filenames_validation = [x for x in positive_filenames if x not in positive_filenames_train]

negative_filenames_train = np.random.choice(negative_filenames, int(len(negative_filenames)*0.8))
negative_filenames_validation = [x for x in negative_filenames if x not in negative_filenames_train]

# If images from the training set are positively diagnosed as diabetic retinopathy, move them to the 'positives' directory. 
# Otherwise, move them to the 'negatives' directory:
for file in os.listdir('../input/train_images/'):
    src = os.path.join('../input/train_images',file)
    if file in positive_filenames:
        if file in positive_filenames_train:
            dst = os.path.join('../data/train/positives',file)
        elif file in positive_filenames_validation:
            dst = os.path.join('../data/validation/positives',file)
    elif file in negative_filenames:
        if file in negative_filenames_train:
            dst = os.path.join('../data/train/negatives',file)
        elif file in negative_filenames_validation:
            dst = os.path.join('../data/validation/negatives',file)
    copyfile(src, dst)

# Define model architecture
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

# Define data generators to convert images to matrices and augment the data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255) # Here, we only rescale the validation data because it doesn't need to be augmented. 
# We are not trying to train it; we are just trying to assess the performance of the model thus far

train_dir = '../data/train'
validation_dir = '../data/validation'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# fit model using our gpu
with tf.device('/gpu:0'):
    history = model2.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#os.mkdir('../models')
# '../models/retinopathy_detector_v2.h5'
model2.save('../retinopathy_detector_v2.h5')
