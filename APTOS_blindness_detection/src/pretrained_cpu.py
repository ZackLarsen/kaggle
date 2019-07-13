from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# In this script, we will be using a pretrained convolutional neural network to 
# try to improve our deep learning model by using features that have already been 
# learned by a model that was trained for much longer on a much larger dataset.

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
train.shape

positives = train[train['diagnosis'] != 0]
negatives = train[train['diagnosis'] == 0]

positive_filenames = list(positives.id_code + '.png')
negative_filenames = list(negatives.id_code + '.png')


# Create train/validation/test folders and positive/negative folders within each:
# Remove whole directory (including all contents in sub-directories) if it exists already:
if 'data' in os.listdir('../'):
    rmtree('../data/')
# The os.mkdir commands below are unnecessary if you use os.makedirs():
#os.mkdir('../data')
#os.mkdir('../data/train')
#os.mkdir('../data/validation')
#os.mkdir('../data/test')
os.makedirs('../data/train/positives')
os.mkdir('../data/train/negatives')
os.makedirs('../data/validation/positives')
os.mkdir('../data/validation/negatives')
os.makedirs('../data/test/positives')
os.mkdir('../data/test/negatives')

# Create a 60/20/20 train/validation/test split. First, get 60% of images for train,
# then put remaining 40% into val_test. Then grab 50% of that (20% overall) for validation.
# Then grab remaining 50% of val_test(20% overall) for test.

# positive_filenames_train = np.random.choice(positive_filenames, floor(int(len(positive_filenames)*0.6)))
# positive_filenames_val_test = [x for x in positive_filenames if x not in positive_filenames_train]
# positive_filenames_validation = np.random.choice(positive_filenames_val_test, floor(int(len(positive_filenames_val_test)*0.5)))
# positive_filenames_test = [x for x in positive_filenames_val_test if x not in positive_filenames_validation]

# negative_filenames_train = np.random.choice(negative_filenames, floor(int(len(negative_filenames)*0.6)))
# negative_filenames_val_test = [x for x in negative_filenames if x not in negative_filenames_train]
# negative_filenames_validation = np.random.choice(negative_filenames_val_test, floor(int(len(negative_filenames_val_test)*0.5)))
# negative_filenames_test = [x for x in negative_filenames_val_test if x not in negative_filenames_validation]

# Hard-coding the sample sizes:
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
    

for dirpath, dirnames, files in os.walk('../data'):
file_count = len(files)
print(f'Files in {dirpath}: {file_count}')


conv_base = VGG16(weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

conv_base.summary()




## We will be recording the output of conv_base on the data and using these outputs as inputs to a new model.

## Start by running instances of the previously introduced ImageDataGenerator to extract images as Numpy arrays as well as their labels. You’ll extract features from these images by calling the predict method of the conv_base model.

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



## Now that we have extracted the features from the train/validation/test splits, we can define the model we want to use and then use it on the newly extracted features.



# Extracted features may need to be flattened to another shape:
train_features = np.reshape(train_features, (2197, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (732, 4 * 4 * 512))
test_features = np.reshape(test_features, (733, 4 * 4 * 512))




# At this point, you can define your densely connected classifier (note the use of dropout
# for regularization) and train it on the data and labels that you just recorded.
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'])

history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels))


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



   
