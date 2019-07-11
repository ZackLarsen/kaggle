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

# Load images into directories according to label and train/test split:
train = pd.read_csv('../input/train.csv')

positives = train[train['diagnosis'] != 0]
negatives = train[train['diagnosis'] == 0]

positive_filenames = list(positives.id_code + '.png')
negative_filenames = list(negatives.id_code + '.png')

# Remove whole directory (including all contents in sub-directories) if it exists already:
if 'data' in os.listdir('../'):
    rmtree('../data/')

os.makedirs('../data/train/positives')
os.mkdir('../data/train/negatives')
os.makedirs('../data/validation/positives')
os.mkdir('../data/validation/negatives')
os.makedirs('../data/test/positives')
os.mkdir('../data/test/negatives')

positive_filenames_train = random.sample(positive_filenames, 1114)
positive_filenames_val_test = [x for x in positive_filenames if x not in positive_filenames_train]
positive_filenames_validation = random.sample(positive_filenames_val_test, 371)
positive_filenames_test = [x for x in positive_filenames_val_test if x not in positive_filenames_validation]

negative_filenames_train = random.sample(negative_filenames, 1083)
negative_filenames_val_test = [x for x in negative_filenames if x not in negative_filenames_train]
negative_filenames_validation = random.sample(negative_filenames_val_test, 361)
negative_filenames_test = [x for x in negative_filenames_val_test if x not in negative_filenames_validation]

# If images from the training set are positively diagnosed as diabetic retinopathy, move them to the 'positives' directory. 
# Otherwise, move them to the 'negatives' directory:
for file in os.listdir('../input/train_images/'):
    src = os.path.join('../input/train_images',file)
    if file in positive_filenames:
        if file in positive_filenames_train:
            dst = os.path.join('../data/train/positives',file)
        elif file in positive_filenames_validation:
            dst = os.path.join('../data/validation/positives',file)
        elif file in positive_filenames_test:
            dst = os.path.join('../data/test/positives',file)
    elif file in negative_filenames:
        if file in negative_filenames_train:
            dst = os.path.join('../data/train/negatives',file)
        elif file in negative_filenames_validation:
            dst = os.path.join('../data/validation/negatives',file)
        elif file in negative_filenames_test:
            dst = os.path.join('../data/test/negatives',file)
    copyfile(src, dst)

conv_base = VGG16(weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))


# Begin extracting features:
base_dir = '../data/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2197)
validation_features, validation_labels = extract_features(validation_dir, 732)
test_features, test_labels = extract_features(test_dir, 733)

# Extracted features may need to be flattened to another shape:
train_features = np.reshape(train_features, (2197, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (732, 4 * 4 * 512))
test_features = np.reshape(test_features, (733, 4 * 4 * 512))

# Add convolutional base model to new model:
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Here, we are FREEZING the weights of the pretrained model because we don't want to lose that information
# while we train the new model
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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

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
