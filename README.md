# CapsNet-vs-CNN

Research focused on comparison between CapsNet and CNN architectures in case of rotated image classification on CIFAR-10 dataset.

## Our Experiments

#### CapsNet with single Digit Layer

- MNIST:

```bash
$ python capsnet/singledigitcapsule/mnist_capsulenet.py
```

- CIFAR10:

```bash
$ python capsnet/singledigitcapsule/cifar_capsulenet.py
```

#### CapsNet with two Digit Layers

- MNIST:

```bash
$ python capsnet/twodigitcapsules/mnist_capsulenet.py
```

- CIFAR10:

```bash
$ python capsnet/twodigitcapsules/cifar_capsulenet.py
```

#### ResNet 110

- MNIST:

```bash
$ python cnn/mnist_resnet.py
```

- CIFAR10:

```bash
$ python cnn/cifar_resnet.py
```

## Results on CIFAR10

### Rotations in training dataset

#### CapsNet with single Digit Layer

Weights are available [here](https://drive.google.com/open?id=1D4dwYZQqVi1nEDJ4wH6SqpNFEcqdXFUG).

```
Rotation: -45 Accuracy: 0.5106
Rotation: -30 Accuracy: 0.6544
Rotation: -15 Accuracy: 0.726
Rotation: 0 Accuracy: 0.7358
Rotation: 15 Accuracy: 0.7192
Rotation: 30 Accuracy: 0.659
Rotation: 45 Accuracy: 0.5224
```

#### CapsNet with two Digit Layers

Weights are available [here](https://drive.google.com/open?id=1ObaKItC4Zn2weHoDGaUyzYSRMNkehepG).

```
Rotation: -45 Accuracy: 0.6207
Rotation: -30 Accuracy: 0.7548
Rotation: -15 Accuracy: 0.8044
Rotation: 0 Accuracy: 0.7985
Rotation: 15 Accuracy: 0.7924
Rotation: 30 Accuracy: 0.746
Rotation: 45 Accuracy: 0.6076
```

#### Resnet 110

Weights are available [here](https://drive.google.com/open?id=1Od3W6AWRANFjnbIndykCHyhFgnETjFP5).

```
Rotation: -45 Accuracy: 0.7697
Rotation: -30 Accuracy: 0.8791
Rotation: -15 Accuracy: 0.9145
Rotation: 0 Accuracy: 0.9076
Rotation: 15 Accuracy: 0.91
Rotation: 30 Accuracy: 0.8829
Rotation: 45 Accuracy: 0.7757
```

### Without rotations in training dataset

#### CapsNet with single Digit Layer

Weights are available [here](https://drive.google.com/open?id=1s4HC3c4ZzwvWb77LPq2J_0vNmiiEMlHW).

```
Rotation: -45 Accuracy: 0.297
Rotation: -30 Accuracy: 0.4019
Rotation: -15 Accuracy: 0.566
Rotation: 0 Accuracy: 0.6805
Rotation: 15 Accuracy: 0.5802
Rotation: 30 Accuracy: 0.4143
Rotation: 45 Accuracy: 0.3
```

#### CapsNet with two Digit Layers

Weights are available [here](https://drive.google.com/open?id=1WwwMXS9R0rXDHyvWczIq_FZjFxiW7L29).

```
Rotation: -45 Accuracy: 0.3966
Rotation: -30 Accuracy: 0.5314
Rotation: -15 Accuracy: 0.7331
Rotation: 0 Accuracy: 0.8144
Rotation: 15 Accuracy: 0.7431
Rotation: 30 Accuracy: 0.536
Rotation: 45 Accuracy: 0.4
```

#### ResNet 110

Weights are available [here](https://drive.google.com/open?id=1ZP_c75dS66UDamOuN_LKLcBxle0yeNuG).

```
Rotation: -45 Accuracy: 0.4226
Rotation: -30 Accuracy: 0.6191
Rotation: -15 Accuracy: 0.87
Rotation: 0 Accuracy: 0.9185
Rotation: 15 Accuracy: 0.8697
Rotation: 30 Accuracy: 0.6332
Rotation: 45 Accuracy: 0.4304
```

## Results on MNIST

Still in progres... :)

## Used model implementations

- [Resnet v2](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) by Keras Team

- [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras) by XifengGuo based upon [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)  
