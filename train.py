import sys
import numpy as np
import random
import gc
import dobble_utils as db

#
# Parameters
#

dir = 'new_dataset'
nrows = 240
ncols = 320
nchannels = 3

card_decks = [
    'exp0-augmented2'
]

nb_card_decks = len(card_decks)

print("")
print("PARAMETERS:")
print("Normalized shape of images :", ncols, " x ", nrows)
print("Card Decks : ", nb_card_decks, card_decks)

#
# Capture images/labels from data set for training and testing
#

train_cards = []
for d in range(0, nb_card_decks):
    train_dir = dir + '/' + card_decks[d]
    filenames = db.get_card_filenames(train_dir)
    print('filenames:', filenames)
    train_cards.append(filenames)

gc.collect()

#
# Read images and pre-process to fixed size
#


train_X = []
train_y = []

for d in range(0, nb_card_decks):
    X, y = db.read_and_process_image(train_cards[d], nrows, ncols)
    train_X.append(np.array(X))
    train_y.append(np.array(y))

train_X = np.concatenate(train_X, axis = 0)
train_y = np.concatenate(train_y, axis = 0)

print('shape train_X:', train_X.shape)
print('shape train_y:', train_y.shape)

print('train_X:', train_X)
print('train_y:', train_y)

print('size train_X:', sys.getsizeof(train_X))
print('size train_y:', sys.getsizeof(train_y))

ntrain = len(train_y)

del train_cards
gc.collect()

#
# Split training data set down into two data sets : training(80%) and validation(20%)
#


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.20,
                                                  random_state = 2)

print("")
print("TRAINING/VALIDATION DATA SETS:")
print("Shape of training data (X) is :", train_X.shape)
print("Shape of training data (y) is :", train_y.shape)
print("Shape of validation data (X) is :", val_X.shape)
print("Shape of validation data (y) is :", val_y.shape)

#
# Create model
#

ntrain = len(train_X)
nval = len(val_X)
batch_size = 16
nepochs = 10

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

# convert labels in range 0-57 to one-hot encoding
train_y = to_categorical(train_y, 30)
val_y = to_categorical(val_y, 30)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (nrows, ncols, nchannels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dense(30))
model.add(layers.Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

print("")
print("MODEL SUMMARY:")
model.summary()

print("")
print("TRAIN MODEL:")

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 360,
    horizontal_flip = True
)
val_datagen = ImageDataGenerator(
    rescale = 1. / 255
)

train_generator = train_datagen.flow(train_X, train_y, batch_size = batch_size)
val_generator = val_datagen.flow(val_X, val_y, batch_size = batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch = int(ntrain / batch_size),
    epochs = nepochs,
    validation_data = val_generator,
    validation_steps = int(nval / batch_size)
)

model.save_weights('dobble_model_weights.h5')
model.save('dobble_model.h5')

#
# Test Model Accuracy
#

model.summary()

test_dir = './dobble_dataset/dobble_test01_cards'
# test_dir = './dobble_dataset/dobble_test02_cards'

test_cards = db.get_card_filenames(test_dir)
random.shuffle(test_cards)

test_X, test_y = db.read_and_process_image(test_cards, ncols, nrows)
del test_cards

ntest = len(test_y)

test_X = np.array(test_X)
test_y = np.array(test_y)

# normalize images
test_X = test_X * (1. / 255)

# convert labels in range 0-57 to one-hot encoding
test_y = to_categorical(test_y, 30)

print("Shape of test data (X) is :", test_X.shape)
print("Shape of test data (y) is :", test_y.shape)

print("")
print("EVALUATE MODEL:")
model.evaluate(test_X, test_y)

test_accuracy = db.test_accuracy(model, ntest, test_X, test_y)
print(test_dir, " : Test Accuracy = ", test_accuracy)