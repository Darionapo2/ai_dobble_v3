# script to test the accuracy of your model and give error bounds
import math
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

import dobble_utils as db

# open the Dobble model
# from keras
model = load_model('dobble_model.h5')
base_dataset_directory = 'dobble_dataset'
nrows = 224
ncols = 224

nchannels = 3

# predict error bounds
test_dir = 'dobble_dataset/dobble_test01_cards'
# test_dir = 'dobble_dataset/dobble_test02_cards'

test_cards = db.get_card_filenames(test_dir)
test_set_size = len(test_cards)

np.random.shuffle(test_cards)

test_X, test_y = db.read_and_process_image(test_cards, nrows, ncols)
# del test_cards

n_test = len(test_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

# element-wise normalization
test_X = test_X * (1.0 / 255)

# convert labels in range 0-57 to one-hot encoding
# from keras
test_y = to_categorical(test_y, 58)

print('Shape of test data (X):', test_X.shape)
print('Shape of test data (y):', test_y.shape)

print('\nEVALUATE MODEL:')
model.evaluate(test_X, test_y)

test_accuracy = db.test_accuracy(model, n_test, test_X, test_y)
print(test_dir, ': Test Accuracy =', test_accuracy)

# analysis of the results
confidence_intervals = [0.5, 0.8, 0.9, 0.95, 0.99]
for conf in confidence_intervals:
    lower_bound, upper_bound = db.get_accuracy_bounds(test_accuracy, test_set_size, conf)
    print(f'{round(conf, 4)} accuracy bound: {round(lower_bound, 4)} - '
          f'{round(upper_bound, 4)}')
