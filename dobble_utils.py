import numpy as np
import cv2
import os
import csv
from collections import OrderedDict

# Capture images and labels from the dataset
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

# Read images and pre-process to fixed size
def read_and_process_image(images, nrows, ncols, labels = True):
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

    if labels:
        return X, y
    else:
        return X

# Load Symbol labels and Card-Symbol mapping
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


# Load Card-Symbol mapping
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
            card_id = card_id + 1

    return mapping

# Test Model Accuracy
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