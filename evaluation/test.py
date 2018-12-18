"""Evaluation script for both CapsNet and CNN models.

Example use:

    $ python evaluation/test.py --network-type=capsnet --weights=./result/weights-01.h5

"""
import argparse

import numpy as np

from dataset import cifar10
from capsnet.capsulenet import CapsNet

CAPSNET_ROUTINGS = 3  # TODO: Allow to override it!


def load_model(network_type, weights):
    if network_type == 'capsnet':
        model, eval_model, _ = CapsNet(input_shape=cifar10.IMAGE_SHAPE, n_class=cifar10.CLASSES, routings=CAPSNET_ROUTINGS)
        model.summary()
        model.load_weights(weights)
        return model
    elif network_type == 'cnn':
        print('Will be done later...')


def evaluate(model):
    if not model:
        print('Model was not loaded...')
        exit(0)
    
    for rotation in [-45, -30, -15, 0, 15, 30, 45]:
        x_test, y_test = cifar10.get_test_data_for_capsnet(rotation)
        y_pred, x_reconstructed = model.predict(x_test, batch_size=100)
        accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test[0], 1))/y_test[0].shape[0]
        print('Rotation:', rotation, 'Accuracy:', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--network-type', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)
    args = parser.parse_args()

    model = load_model(args.network_type, args.weights)
    evaluate(model)

