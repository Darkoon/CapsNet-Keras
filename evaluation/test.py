"""Evaluation script for both CapsNet and CNN models.

Example use:

    $ python evaluation/test.py --dataset=cifar --network-type=capsnet_single_digit_capsule --weights=./weights-01.h5

"""
import argparse

import numpy as np

from dataset import cifar10

CAPSNET_ROUTINGS = 3  # TODO: Allow to override it!


def evaluate(dataset, network_type, weights):
    if dataset == 'cifar':
        from dataset import cifar10
        dataset_module = cifar10
    elif dataset == 'mnist':
        from dataset import mnist
        dataset_module = mnist
    else:
        print('ERROR! Unsupported dataset!')
        exit(1)

    if 'capsnet' in network_type:
        if dataset == 'mnist' and network_type == 'capsnet_single_digit_capsule':
            from capsnet.singledigitcapsule.mnist_capsulenet import CapsNet
            model, _, _ = CapsNet(input_shape=mnist.IMAGE_SHAPE, n_class=mnist.CLASSES, routings=CAPSNET_ROUTINGS)
        elif dataset == 'mnist' and network_type == 'capsnet_two_digits_capsules':
            from capsnet.twodigitcapsules.mnist_capsulenet import CapsNet
            model, _, _ = CapsNet(input_shape=mnist.IMAGE_SHAPE, n_class=mnist.CLASSES, routings=CAPSNET_ROUTINGS)
        if dataset == 'cifar' and network_type == 'capsnet_single_digit_capsule':
            from capsnet.singledigitcapsule.cifar_capsulenet import CapsNet
            model, _, _ = CapsNet(input_shape=cifar10.IMAGE_SHAPE, n_class=cifar10.CLASSES, routings=CAPSNET_ROUTINGS)
        elif dataset == 'cifar' and network_type == 'capsnet_two_digits_capsules':
            from capsnet.twodigitcapsules.cifar_capsulenet import CapsNet
            model, _, _ = CapsNet(input_shape=cifar10.IMAGE_SHAPE, n_class=cifar10.CLASSES, routings=CAPSNET_ROUTINGS)
        else:
            print('ERROR! Unsupported CapsNet type!')
            exit(1)
    elif network_type == 'cnn':
        from cnn.resnet import resnet_v2
        model = resnet_v2(input_shape=cifar10.IMAGE_SHAPE, depth=110)
    else:
        print('ERROR! Unsupported Network Type!')
        exit(1)

    model.summary()
    model.load_weights(weights)

    for rotation in [-45, -30, -15, 0, 15, 30, 45]:
        if 'capsnet' in network_type:
            x_test, y_test = dataset_module.get_test_data_for_capsnet(rotation)
            y_pred, x_reconstructed = model.predict(x_test, batch_size=100)
            accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test[0], 1))/y_test[0].shape[0]
        elif network_type == 'cnn':
            x_test, y_test = dataset_module.get_test_data_for_cnn(rotation)
            scores = model.evaluate(x_test, y_test, verbose=1)
            accuracy = scores[1]

        print('Rotation:', rotation, 'Accuracy:', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of CNN and CapsNet networks against rotated images.')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--network-type', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)
    args = parser.parse_args()
    print('args:', args)

    evaluate(args.dataset, args.network_type, args.weights)

