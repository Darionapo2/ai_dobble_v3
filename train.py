import gc, os
import numpy as np
import dobble_utils as db
from keras import layers, models
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

os.environ['KERAS_BACKEND'] = 'tensorflow'

dataset_dir = 'new_dataset'
nrows = 240
ncols = 320
nchannels = 3

card_decks = ['exp0-augmented2']

nb_card_decks = len(card_decks)


def main():
    train_cards = prepare_train_cards(card_decks)
    X, y, n_train = build_dataset(train_cards)

    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)
    print('Training dataset overall number of samples', n_train)

    seed = 3
    test_size = 0.20

    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size = test_size, random_state = seed
    )

    print('Shape of training data (train_X):', train_X.shape)
    print('Shape of training data (train_y):', train_y.shape)
    print('Shape of validation data (val_X):', val_X.shape)
    print('Shape of validation data (val_y):', val_y.shape)

    # convert labels in range 0-29 to one-hot encoding
    train_y = to_categorical(train_y, 30)
    val_y = to_categorical(val_y, 30)

    model = create_model()

    entrenador_automatico(
        model, train_X, train_y, val_X, val_y,
        batch_size = 16, nepochs = 10
    )


# Capture images/labels from data set for training and testing
def prepare_train_cards(decks):
    train_cards = []
    for deck in decks:
        train_dir = f'{dataset_dir}/{deck}'
        filenames = db.get_card_filenames(train_dir)
        train_cards.append(filenames)

    gc.collect()
    return train_cards


# Read images and pre-process to fixed size
def build_dataset(train_cards):
    all_X = []
    all_y = []

    for d in range(0, nb_card_decks):
        X, y = db.read_and_process_image(train_cards[d], nrows, ncols)
        all_X.append(np.array(X))
        all_y.append(np.array(y))

    all_X = np.concatenate(all_X, axis = 0)
    all_y = np.concatenate(all_y, axis = 0)
    n_train = len(all_y)

    del train_cards
    gc.collect()

    return all_X, all_y, n_train


def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(
        filters = 32, kernel_size = (3, 3),
        activation = 'relu', input_shape = (nrows, ncols, nchannels))

    )
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Flatten())

    model.add(layers.Dense(units = 512))
    model.add(layers.Activation(activation = 'relu'))

    model.add(layers.Dense(units = 29))
    model.add(layers.Activation(activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    model.summary()

    return model


def entrenador_automatico(model, train_X, train_y, val_X, val_y, batch_size, nepochs):
    ntrain = len(train_X)
    nval = len(val_X)

    train_datagen = ImageDataGenerator(
        rescale = 1.0 / 255,
        rotation_range = 360,
        horizontal_flip = True
    )

    val_datagen = ImageDataGenerator(rescale = 1.0 / 255)

    # Definition of useful objects implemented in the training
    train_generator = train_datagen.flow(train_X, train_y, batch_size = batch_size)
    val_generator = val_datagen.flow(val_X, val_y, batch_size = batch_size)

    # The actual model training
    history = model.fit(
        train_generator,
        steps_per_epoch = int(ntrain / batch_size),
        epochs = nepochs,
        validation_data = val_generator,
        validation_steps = int(nval / batch_size)
    )

    model.save_weights('dobble_model_weights.h5')
    model.save('dobble_model.h5')

    return history

if __name__ == '__main__':
    main()