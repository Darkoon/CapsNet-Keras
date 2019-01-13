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

**BELOW VALUES WERE NOT PROCESSED YET!** We will put them into tables & graphs, don't worry! :)

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

### Rotations in training dataset

#### CapsNet with single Digit Layer

Weights are available [here](https://drive.google.com/open?id=1demuj6N_8PTQWs_vP0hvHFnCSRx8Udx3).

```
Rotation: -180 Accuracy: 0.4261
Rotation: -165 Accuracy: 0.4236
Rotation: -150 Accuracy: 0.4247
Rotation: -135 Accuracy: 0.4011
Rotation: -120 Accuracy: 0.3427
Rotation: -105 Accuracy: 0.2936
Rotation: -90 Accuracy: 0.3306
Rotation: -75 Accuracy: 0.5177
Rotation: -60 Accuracy: 0.802
Rotation: -45 Accuracy: 0.9582
Rotation: -30 Accuracy: 0.9901
Rotation: -15 Accuracy: 0.993
Rotation: 0 Accuracy: 0.9929
Rotation: 15 Accuracy: 0.9919
Rotation: 30 Accuracy: 0.989
Rotation: 45 Accuracy: 0.9433
Rotation: 60 Accuracy: 0.6914
Rotation: 75 Accuracy: 0.3606
Rotation: 90 Accuracy: 0.2284
Rotation: 105 Accuracy: 0.2705
Rotation: 120 Accuracy: 0.3785
Rotation: 135 Accuracy: 0.4427
Rotation: 150 Accuracy: 0.4572
Rotation: 165 Accuracy: 0.444
```

#### CapsNet with two Digit Layers

Weights are available [here](https://drive.google.com/open?id=1ODsS__IHprAVFhQpOiKCT3u6TqcwfLRG).

```
Rotation: -180 Accuracy: 0.4274
Rotation: -165 Accuracy: 0.443
Rotation: -150 Accuracy: 0.4343
Rotation: -135 Accuracy: 0.4008
Rotation: -120 Accuracy: 0.3155
Rotation: -105 Accuracy: 0.2448
Rotation: -90 Accuracy: 0.2548
Rotation: -75 Accuracy: 0.4569
Rotation: -60 Accuracy: 0.7752
Rotation: -45 Accuracy: 0.944
Rotation: -30 Accuracy: 0.9836
Rotation: -15 Accuracy: 0.9899
Rotation: 0 Accuracy: 0.9871
Rotation: 15 Accuracy: 0.9913
Rotation: 30 Accuracy: 0.9883
Rotation: 45 Accuracy: 0.9483
Rotation: 60 Accuracy: 0.7153
Rotation: 75 Accuracy: 0.3844
Rotation: 90 Accuracy: 0.235
Rotation: 105 Accuracy: 0.2653
Rotation: 120 Accuracy: 0.3577
Rotation: 135 Accuracy: 0.423
Rotation: 150 Accuracy: 0.4478
Rotation: 165 Accuracy: 0.4369
```

#### Resnet 110

Weights are available [here](https://drive.google.com/open?id=1AUH88BcE8OUE5rg9PdxNlnbXjo7Zh6Z7).

```
Rotation: -180 Accuracy: 0.4216
Rotation: -165 Accuracy: 0.4307
Rotation: -150 Accuracy: 0.4305
Rotation: -135 Accuracy: 0.4181
Rotation: -120 Accuracy: 0.3825
Rotation: -105 Accuracy: 0.3649
Rotation: -90 Accuracy: 0.4119
Rotation: -75 Accuracy: 0.6214
Rotation: -60 Accuracy: 0.8523
Rotation: -45 Accuracy: 0.9673
Rotation: -30 Accuracy: 0.9901
Rotation: -15 Accuracy: 0.9951
Rotation: 0 Accuracy: 0.9956
Rotation: 15 Accuracy: 0.9961
Rotation: 30 Accuracy: 0.9926
Rotation: 45 Accuracy: 0.9654
Rotation: 60 Accuracy: 0.7894
Rotation: 75 Accuracy: 0.496
Rotation: 90 Accuracy: 0.337
Rotation: 105 Accuracy: 0.3524
Rotation: 120 Accuracy: 0.3983
Rotation: 135 Accuracy: 0.4316
Rotation: 150 Accuracy: 0.4296
Rotation: 165 Accuracy: 0.4226
```

### Without rotations in training dataset

Still in progress... :)

## Used model implementations

- [Resnet v2](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) by Keras Team

- [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras) by XifengGuo based upon [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)  
