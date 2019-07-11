import os
import numpy as np
import pandas as pd
from math import floor
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from shutil import copyfile, rmtree
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv')

## Add the .png file extension to the filenames and make the diagnosis a string instead of integer.
split_df = train
split_df['filename'] = [x + '.png' for x in split_df.id_code]
split_df['diagnosis_str'] = split_df['diagnosis'].astype(str)

## Perform 60/20/20 train/val/test split
train_df = train.iloc[:floor(len(train) * 0.6),]
validation_df = train.iloc[floor(len(train) * 0.6):floor(len(train) * 0.8),]
test_df = train.iloc[floor(len(train) * 0.8):,]

## Load VGG16 pretrained model
conv_base = VGG16(weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

## Add convolutional base model to new model:
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

## Here, we are FREEZING the weights of the pretrained model because we don't want to lose that information
## while we train the new model
conv_base.trainable = False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory='../input/train_images/', 
    x_col='filename', 
    y_col='diagnosis_str',
    target_size=(150, 150), 
    color_mode='rgb', 
    classes=None, 
    class_mode='categorical', 
    batch_size=20)

validation_generator = train_datagen.flow_from_dataframe(
    validation_df, 
    directory='../input/train_images/', 
    x_col='filename', 
    y_col='diagnosis_str', 
    target_size=(150, 150), 
    color_mode='rgb', 
    classes=None, 
    class_mode='categorical', 
    batch_size=20)

## Compile the model
model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

## Train the model for 30 epochs
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

## Plot training and validation accuracy / loss:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

## Save model
model.save('APTOS_vgg16.h5')
