import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
import dobble_utils as db

model = load_model('dobble_model7.h5')
base_dataset_directory = 'dobble_dataset'

nrows = 320
ncols = 240
nchannels = 3

def main():

    common_symbol = jugador_de_dobble(
        'new_dataset/exp0/05/05_card_exp0.jpg',
        'new_dataset/exp0/06/06_card_exp0.jpg'
    )

    print('The symbol in common in the 2 given cards is:', common_symbol)

    test_set_size, shape_test_X, shape_test_y, accuracy = test_accuracy('new_dataset/exp0')
    print('Accuracy:', accuracy)

def test_accuracy(test_dir):
    test_cards = db.get_card_filenames(test_dir)
    test_set_size = len(test_cards)

    np.random.shuffle(test_cards)

    test_X, test_y = db.read_and_process_image(test_cards, nrows, ncols, labels = True)
    # All labels are shifted of 1 number because we want the first class (card 1) to be 0.
    test_y = np.subtract(test_y, 1)
    del test_cards

    n_test = len(test_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # element-wise normalization
    test_X = test_X / 255.0

    test_y = to_categorical(test_y, 29)

    model.evaluate(test_X, test_y)
    accuracy = db.test_accuracy(model, n_test, test_X, test_y)

    return test_set_size, test_X.shape, test_y.shape, accuracy

def classify(card_dir):
    image_X = db.read_and_process_image(
        [card_dir],
        nrows, ncols,
        labels = False
    )

    card_x = np.array([image_X[0]])

    prediction = model.predict(card_x)[0]
    result = list(prediction).index(max(prediction))

    return result

def jugador_de_dobble(card1_dir, card2_dir):
    mapping = db.load_card_symbol_mapping('new_dataset/cards_symbols_mapping.csv')
    labels = db.load_symbol_labels('new_dataset/deck_symbols.csv')

    class_card1 = classify(card1_dir)
    class_card2 = classify(card2_dir)

    v1 = set(mapping[class_card1 + 1])
    v2 = set(mapping[class_card2 + 1])

    symbol_id = int(list(v1 & v2)[0])

    return labels[symbol_id]

if __name__ == '__main__':
    main()

