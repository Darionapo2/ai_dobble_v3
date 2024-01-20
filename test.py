import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
import dobble_utils as db

model = load_model('dobble_model.h5')
base_dataset_directory = 'dobble_dataset'

nrows = 320
ncols = 240
nchannels = 3

def main():
    jugador_de_dobble('new_dataset/test/test_cards/01_card_exp0.jpg',
                      'new_dataset/test/test_cards/09_card_exp0.jpg')

def test_accuracy(test_dir):
    test_cards = db.get_card_filenames(test_dir)
    test_set_size = len(test_cards)

    np.random.shuffle(test_cards)

    test_X, test_y = db.read_and_process_image(test_cards, nrows, ncols, labels = True)
    del test_cards

    n_test = len(test_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # element-wise normalization
    test_X = test_X / 255.0

    test_y = to_categorical(test_y, 30)

    model.evaluate(test_X, test_y)
    accuracy = db.test_accuracy(model, n_test, test_X, test_y)

    return test_set_size, test_X.shape, test_y.shape, accuracy

def jugador_de_dobble(card1_dir, card2_dir):
    mapping = db.load_card_symbol_mapping('new_dataset/cards_symbols_mapping.csv')
    labels = db.load_symbol_labels('new_dataset/deck_symbols.csv')

    images_X = db.read_and_process_image(
        [card1_dir, card2_dir],
        nrows, ncols,
        labels = False)

    card1_x = np.array([images_X[0]])
    card2_x = np.array([images_X[1]])

    prediction1 = model.predict(card1_x)[0]
    prediction2 = model.predict(card2_x)[0]

    result1 = list(prediction1).index(max(prediction1))
    result2 = list(prediction2).index(max(prediction2))

    v1 = set(mapping[result1])
    v2 = set(mapping[result2])

    symbol_id = int(list(v1 & v2)[0])

    return labels[symbol_id]

if __name__ == '__main__':
    main()

