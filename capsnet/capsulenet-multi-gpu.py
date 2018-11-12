"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet-multi-gpu.py
       python capsulenet-multi-gpu.py --gpus 2
       ... ...

Result:
    About 55 seconds per epoch on two GTX1080Ti GPU cards

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import os
import argparse

import tensorflow as tf
from tensorflow import keras
import numpy as np

from dataset import mnist
from capsnet.capsulenet import CapsNet, margin_loss, manipulate_latent, test
from capsnet.utils import plot_log

keras.backend.set_image_data_format('channels_last')


def train(model, args):
    """Training a CapsuleNet.

    :param model: the CapsuleNet model
    :param args: arguments
    :return: The trained model
    """
    # Setup callbacks
    log = keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = keras.callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                     batch_size=args.batch_size, histogram_freq=args.debug)
    lr_decay = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon])
    
    # Training
    model.fit_generator(generator=mnist.get_train_generator(args.batch_size),
                        steps_per_epoch=int(mnist.TEST_SIZE / args.batch_size),
                        epochs=args.epochs,
                        validation_data=mnist.get_validation_data(),
                        callbacks=[log, tb, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    plot_log(args.save_dir + '/log.csv', show=True)

    return model


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--gpus', default=2, type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    with tf.device('/cpu:0'):
        model, eval_model, manipulate_model = CapsNet(input_shape=mnist.IMAGE_SHAPE,
                                                      n_class=mnist.CLASSES,
                                                      routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        # define muti-gpu model
        multi_model = keras.utils.multi_gpu_model(model, gpus=args.gpus)
        train(model=multi_model, args=args)
        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
        test(model=eval_model, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
