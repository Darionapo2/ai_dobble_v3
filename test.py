# model designed to perform the specific task of identifying and comparing symbols on game cards

import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
import dobble_utils as db

# we use the Keras library to load the pre-trained neural network model and evaluate its
# performance on a test data set.
model = load_model('dobble_model7.h5')
base_dataset_directory = 'dobble_dataset'

# number of lines and columns of pixels in the picture (will be scaled this way)
nrows = 320
ncols = 240
nchannels = 3 # colour channels in images

def main():

    common_symbol = jugador_de_dobble(
        'new_dataset/exp0/05/05_card_exp0.jpg',
        'new_dataset/exp0/06/06_card_exp0.jpg'
    )

    print('The symbol in common in the 2 given cards is:', common_symbol)

    # test the accuracy of our trained model
    test_set_size, shape_test_X, shape_test_y, accuracy = test_accuracy('new_dataset/exp0')
    print('Accuracy:', accuracy)

def test_accuracy(test_dir):
    # returns a list of images file paths.
    test_cards = db.get_card_filenames(test_dir)
    test_set_size = len(test_cards)

    # randomly sort the cards to prevenet any possible 'bias' in the evaluation of the
    # classication performances
    np.random.shuffle(test_cards)

    # divide labels and actual dataset
    test_X, test_y = db.read_and_process_image(test_cards, nrows, ncols, labels = True)
    # All labels are shifted of 1 number because we want the first class (card 1) to be 0.
    test_y = np.subtract(test_y, 1)
    del test_cards

    n_test = len(test_y) # number of records in the dataset
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # element-wise normalization
    test_X = test_X / 255.0

    # perform a one-hot encoding of the labels, maps every possible label value with binary lists
    test_y = to_categorical(test_y, 29)

    # evaluation of the ML model
    model.evaluate(test_X, test_y)
    accuracy = db.test_accuracy(model, n_test, test_X, test_y)

    return test_set_size, test_X.shape, test_y.shape, accuracy

# return the string associated with the classified record (label obtained)
def classify(card_dir):
    image_X = db.read_and_process_image(
        [card_dir],
        nrows, ncols,
        labels = False
    )

    card_x = np.array([image_X[0]])

    prediction = model.predict(card_x)[0]

    # we want to get the value closest to 1 in the classification set.
    # the index of this value rappresent the card that the model has classified.
    result = list(prediction).index(max(prediction))
    return result

def jugador_de_dobble(card1_dir, card2_dir):
    # loads a mapping between the card classes and the corresponding symbols
    mapping = db.load_card_symbol_mapping('new_dataset/cards_symbols_mapping.csv')
    # load symbol labels
    labels = db.load_symbol_labels('new_dataset/deck_symbols.csv')

    # obtain the predicted class for the two given cards
    class_card1 = classify(card1_dir)
    class_card2 = classify(card2_dir)

    # sets of symbols associated with the predicted classes
    v1 = set(mapping[class_card1 + 1])
    v2 = set(mapping[class_card2 + 1])

    # finds the intersection of the sets --> this will be the symbol in common between the
    # predicted cards
    symbol_id = int(list(v1 & v2)[0])

    # return the corrisponding string in the symbols table
    return labels[symbol_id]

if __name__ == '__main__':
    main()

