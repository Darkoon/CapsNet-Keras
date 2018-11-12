from tensorflow import keras
import numpy as np

MNIST_IMAGE_SHAPE = (28, 28, 1)
MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000
MNIST_CLASSES = 10


# def _load_mnist():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
#     x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
#     x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
#
#     mean_image = np.mean(x_train, axis=1)
#     std_image = np.std(x_train, axis=1)
#
#     x_train = (x_train - mean_image) / std_image
#     x_test = (x_test - mean_image) / std_image
#
#     y_train = keras.utils.to_categorical(y_train.astype('float32'))
#     y_test = keras.utils.to_categorical(y_test.astype('float32'))
# 
#     return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

mean_image = np.mean(x_train, axis=0)
std_image = np.std(x_train, axis=0)

x_train = np.nan_to_num((x_train - mean_image) / std_image)
x_test = np.nan_to_num((x_test - mean_image) / std_image)

y_train = keras.utils.to_categorical(y_train.astype('float32'))
y_test = keras.utils.to_categorical(y_test.astype('float32'))


def get_mnist_train_generator(batch_size):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.0,
                                                                 height_shift_range=0.0)
    generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])


def get_mnist_validation_data():
    return [[x_test, y_test], [y_test, x_test]]


def get_mnist_test_data():
    return (x_test, y_test)

