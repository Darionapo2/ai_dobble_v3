import math

import numpy as np
import cv2

import os

import csv
from collections import OrderedDict


#
# Capture images/labels from data set for training and testing
#

def get_card_filenames(directory_name):
    # get the list of sub-folders for the selected directory
    dirs = sorted(os.listdir(directory_name))
    # build a list of subdirectories with a fixed prefix and the changing sub-folder
    subdirs = [f'{directory_name}/{d}' for d in dirs]

    cards = []

    # add all cards of the dataset in the same list "cards"
    for i, subdir in enumerate(subdirs):
        cards_filenames = os.listdir(subdir)

        # sometimes there are more than just 1 card for each sub-dir
        cards.extend([f'{subdir}/{card}' for card in cards_filenames])

    return cards


#
# Read images and pre-process to fixed size
#

def read_and_process_image(images, nrows, ncols):
    X = []
    y = []

    # iterate trought the list of images specified
    for i, image_filename in enumerate(images):
        im = cv2.imread(image_filename, cv2.IMREAD_COLOR)

        # perform a resize of the image according to the specified dimentions
        resized_im = cv2.resize(im, (nrows, ncols), interpolation = cv2.INTER_CUBIC)
        X.append(resized_im)

        # get the integer rappresenting the number of dataset to test.
        # example: ['dobble_dataset', 'dobble_test01_cards', '02', 'card02_00.tif'] --> 2
        # and add it into the "y" testing variable
        y_lst = image_filename.split('/')
        y.append(int(y_lst[-2]))

    return X, y


#
# Create collage for cards 01-55 (5 rows, 11 columns)
#

def create_collage(cards_X, cards_y):
    cards_idx = np.where(np.logical_and(cards_y >= 1, cards_y <= 55))
    cards_55 = cards_X[cards_idx]

    h, w, z = cards_X[0, :, :, :].shape
    w11 = w * 11
    h5 = h * 5
    collage = np.zeros((h5, w11, 3), np.uint8)
    idx = 0
    for r in range(0, 5):
        for c in range(0, 11):
            collage[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = cards_55[idx, :, :, :]
            idx = idx + 1

    return collage


#
# Load Symbol labels and Card-Symbol mapping
#

def load_symbol_labels(symbol_filename):
    symbols = OrderedDict()
    with open(symbol_filename, 'r') as file:
        reader = csv.reader(file)
        symbol_id = 1
        for row in reader:
            symbol_label = row[1]
            symbols[symbol_id] = symbol_label

            symbol_id = symbol_id + 1
    return symbols


#
# Load Card-Symbol mapping
#

def load_card_symbol_mapping(mapping_filename):
    mapping = OrderedDict()
    with open(mapping_filename, 'r') as file:
        reader = csv.reader(file)
        card_id = 1
        for row in reader:
            card_mapping = []
            for i, val in enumerate(row[1:]):
                if val == '1':
                    card_mapping.append(i + 1)
            mapping[card_id] = card_mapping
            #
            card_id = card_id + 1

    return mapping


#
# Test Model Accuracy
#

def test_accuracy(model, test_n, test_X, test_y):
    ntotal = 0
    ncorrect = 0
    predictions = model.predict(test_X)
    for i in range(test_n):
        y = test_y[i, :]
        pred = predictions[i, :]
        max_y = np.argmax(y)
        max_pred = np.argmax(pred)
        ntotal += 1
        if max_pred == max_y:
            ncorrect += 1
    return ncorrect / ntotal


def get_accuracy_bounds(mean, sample_size, confidence):
    _z_values = {0.5: 0.67, 0.68: 1.0, 0.8: 1.28, 0.9: 1.64, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}

    if mean < 0.0 or mean > 1.0:
        raise UserWarning('mean must be between 0 and 1')
    if sample_size <= 0:
        raise UserWarning('sampleSize should be positive')

    # lookup the Z value depending on the confidence
    zvalue = _z_values.get(confidence)

    # get the standard deviation
    stdev = math.sqrt((mean * (1 - mean)) / sample_size)

    # multiply the standard deviation by the zvalue
    interval = zvalue * stdev

    lower_bound = mean - interval
    upper_bound = mean + interval

    return lower_bound, upper_bound
