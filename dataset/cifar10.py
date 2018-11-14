import functools

from tensorflow import keras
import numpy as np


IMAGE_SHAPE = (32, 32, 3)
TRAIN_SIZE = 50000
TEST_SIZE = 10000
CLASSES = 10


@functools.lru_cache()
def _load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

    y_train = keras.utils.to_categorical(y_train.astype('float32'))
    y_test = keras.utils.to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def get_train_generator_for_cnn(batch_size):
    (x_train, y_train), (_, _) = _load_data()

    config = {
        'rotation_range': 30,  # Random rotations from -30 deg to 30 deg
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': False,  # Doesn't make sense in CIFAR10
    }
    train_datagen = keras.preprocessing.image.ImageDataGenerator(**config)
    generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    while 1:
        x_batch, y_batch = generator.next()
        yield (x_batch, y_batch)


def get_train_generator_for_capsnet(batch_size):
    for (x_batch, y_batch) in get_train_generator_for_cnn(batch_size):
        yield ([x_batch, y_batch], [y_batch, x_batch])


def get_validation_data_for_cnn():
    (_, _), (x_test, y_test) = _load_data()
    return [x_test, y_test]


def get_validation_data_for_capsnet():
    x_test, y_test = get_validation_data_for_cnn()
    return [[x_test, y_test], [y_test, x_test]]


def get_test_data():
    (_, _), (x_test, y_test) = _load_data()
    return (x_test, y_test)

