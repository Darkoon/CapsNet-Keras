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
Rotation: -180 Accuracy: 0.3316
Rotation: -165 Accuracy: 0.3184
Rotation: -150 Accuracy: 0.3115
Rotation: -135 Accuracy: 0.29
Rotation: -120 Accuracy: 0.2629
Rotation: -105 Accuracy: 0.2661
Rotation: -90 Accuracy: 0.2873
Rotation: -75 Accuracy: 0.3111
Rotation: -60 Accuracy: 0.3774
Rotation: -45 Accuracy: 0.5106
Rotation: -30 Accuracy: 0.6544
Rotation: -15 Accuracy: 0.726
Rotation: 0 Accuracy: 0.7358
Rotation: 15 Accuracy: 0.7192
Rotation: 30 Accuracy: 0.659
Rotation: 45 Accuracy: 0.5224
Rotation: 60 Accuracy: 0.3749
Rotation: 75 Accuracy: 0.3031
Rotation: 90 Accuracy: 0.2793
Rotation: 105 Accuracy: 0.2569
Rotation: 120 Accuracy: 0.2545
Rotation: 135 Accuracy: 0.2651
Rotation: 150 Accuracy: 0.2978
Rotation: 165 Accuracy: 0.3134
```

#### CapsNet with two Digit Layers

Weights are available [here](https://drive.google.com/open?id=1ObaKItC4Zn2weHoDGaUyzYSRMNkehepG).

```
Rotation: -180 Accuracy: 0.337
Rotation: -165 Accuracy: 0.3283
Rotation: -150 Accuracy: 0.323
Rotation: -135 Accuracy: 0.2941
Rotation: -120 Accuracy: 0.299
Rotation: -105 Accuracy: 0.3152
Rotation: -90 Accuracy: 0.3315
Rotation: -75 Accuracy: 0.3703
Rotation: -60 Accuracy: 0.457
Rotation: -45 Accuracy: 0.6071
Rotation: -30 Accuracy: 0.7463
Rotation: -15 Accuracy: 0.8006
Rotation: 0 Accuracy: 0.791
Rotation: 15 Accuracy: 0.7965
Rotation: 30 Accuracy: 0.7504
Rotation: 45 Accuracy: 0.6058
Rotation: 60 Accuracy: 0.4494
Rotation: 75 Accuracy: 0.3682
Rotation: 90 Accuracy: 0.3285
Rotation: 105 Accuracy: 0.3053
Rotation: 120 Accuracy: 0.298
Rotation: 135 Accuracy: 0.2968
Rotation: 150 Accuracy: 0.3244
Rotation: 165 Accuracy: 0.3346
```

#### Resnet 110

Weights are available [here](https://drive.google.com/open?id=1Od3W6AWRANFjnbIndykCHyhFgnETjFP5).

```
Rotation: -180 Accuracy: 0.4162
Rotation: -165 Accuracy: 0.4185
Rotation: -150 Accuracy: 0.4171
Rotation: -135 Accuracy: 0.3926
Rotation: -120 Accuracy: 0.3847
Rotation: -105 Accuracy: 0.3929
Rotation: -90 Accuracy: 0.4284
Rotation: -75 Accuracy: 0.4992
Rotation: -60 Accuracy: 0.6246
Rotation: -45 Accuracy: 0.7697
Rotation: -30 Accuracy: 0.8791
Rotation: -15 Accuracy: 0.9145
Rotation: 0 Accuracy: 0.9076
Rotation: 15 Accuracy: 0.91
Rotation: 30 Accuracy: 0.8829
Rotation: 45 Accuracy: 0.7757
Rotation: 60 Accuracy: 0.6423
Rotation: 75 Accuracy: 0.5137
Rotation: 90 Accuracy: 0.4459
Rotation: 105 Accuracy: 0.4049
Rotation: 120 Accuracy: 0.3834
Rotation: 135 Accuracy: 0.384
Rotation: 150 Accuracy: 0.4127
Rotation: 165 Accuracy: 0.4217
```

### Without rotations in training dataset

#### CapsNet with single Digit Layer

Weights are available [here](https://drive.google.com/open?id=1s4HC3c4ZzwvWb77LPq2J_0vNmiiEMlHW).

```
Rotation: -180 Accuracy: 0.2729
Rotation: -165 Accuracy: 0.2431
Rotation: -150 Accuracy: 0.22
Rotation: -135 Accuracy: 0.2042
Rotation: -120 Accuracy: 0.1959
Rotation: -105 Accuracy: 0.2006
Rotation: -90 Accuracy: 0.249
Rotation: -75 Accuracy: 0.2437
Rotation: -60 Accuracy: 0.2559
Rotation: -45 Accuracy: 0.297
Rotation: -30 Accuracy: 0.4019
Rotation: -15 Accuracy: 0.566
Rotation: 0 Accuracy: 0.6805
Rotation: 15 Accuracy: 0.5802
Rotation: 30 Accuracy: 0.4143
Rotation: 45 Accuracy: 0.3
Rotation: 60 Accuracy: 0.2462
Rotation: 75 Accuracy: 0.2345
Rotation: 90 Accuracy: 0.2465
Rotation: 105 Accuracy: 0.2083
Rotation: 120 Accuracy: 0.1897
Rotation: 135 Accuracy: 0.2
Rotation: 150 Accuracy: 0.2146
Rotation: 165 Accuracy: 0.2382
```

#### CapsNet with two Digit Layers

Weights are available [here](https://drive.google.com/open?id=1WwwMXS9R0rXDHyvWczIq_FZjFxiW7L29).

```
Rotation: -180 Accuracy: 0.3493
Rotation: -165 Accuracy: 0.3259
Rotation: -150 Accuracy: 0.2809
Rotation: -135 Accuracy: 0.2537
Rotation: -120 Accuracy: 0.2461
Rotation: -105 Accuracy: 0.2624
Rotation: -90 Accuracy: 0.304
Rotation: -75 Accuracy: 0.3043
Rotation: -60 Accuracy: 0.3227
Rotation: -45 Accuracy: 0.3966
Rotation: -30 Accuracy: 0.5314
Rotation: -15 Accuracy: 0.7331
Rotation: 0 Accuracy: 0.8144
Rotation: 15 Accuracy: 0.7431
Rotation: 30 Accuracy: 0.536
Rotation: 45 Accuracy: 0.4
Rotation: 60 Accuracy: 0.3309
Rotation: 75 Accuracy: 0.3023
Rotation: 90 Accuracy: 0.3082
Rotation: 105 Accuracy: 0.259
Rotation: 120 Accuracy: 0.236
Rotation: 135 Accuracy: 0.2406
Rotation: 150 Accuracy: 0.2553
Rotation: 165 Accuracy: 0.3046
```

#### ResNet 110

Weights are available [here](https://drive.google.com/open?id=1ZP_c75dS66UDamOuN_LKLcBxle0yeNuG).

```
Rotation: -180 Accuracy: 0.4162
Rotation: -165 Accuracy: 0.3578
Rotation: -150 Accuracy: 0.2859
Rotation: -135 Accuracy: 0.2569
Rotation: -120 Accuracy: 0.2769
Rotation: -105 Accuracy: 0.3288
Rotation: -90 Accuracy: 0.3989
Rotation: -75 Accuracy: 0.3905
Rotation: -60 Accuracy: 0.3864
Rotation: -45 Accuracy: 0.4226
Rotation: -30 Accuracy: 0.6191
Rotation: -15 Accuracy: 0.87
Rotation: 0 Accuracy: 0.9185
Rotation: 15 Accuracy: 0.8697
Rotation: 30 Accuracy: 0.6332
Rotation: 45 Accuracy: 0.4304
Rotation: 60 Accuracy: 0.3909
Rotation: 75 Accuracy: 0.3863
Rotation: 90 Accuracy: 0.4138
Rotation: 105 Accuracy: 0.3345
Rotation: 120 Accuracy: 0.2717
Rotation: 135 Accuracy: 0.2482
Rotation: 150 Accuracy: 0.2858
Rotation: 165 Accuracy: 0.3536
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

#### CapsNet with single Digit Layer

Weights are available [here](https://drive.google.com/open?id=1s4KzwmXSCywwWQVQtfZbFd50_9ml4zoJ).

```
Rotation: -180 Accuracy: 0.4408
Rotation: -165 Accuracy: 0.4346
Rotation: -150 Accuracy: 0.4049
Rotation: -135 Accuracy: 0.3402
Rotation: -120 Accuracy: 0.2597
Rotation: -105 Accuracy: 0.1881
Rotation: -90 Accuracy: 0.1733
Rotation: -75 Accuracy: 0.2488
Rotation: -60 Accuracy: 0.3991
Rotation: -45 Accuracy: 0.644
Rotation: -30 Accuracy: 0.9004
Rotation: -15 Accuracy: 0.9829
Rotation: 0 Accuracy: 0.9932
Rotation: 15 Accuracy: 0.9815
Rotation: 30 Accuracy: 0.8631
Rotation: 45 Accuracy: 0.5472
Rotation: 60 Accuracy: 0.2637
Rotation: 75 Accuracy: 0.147
Rotation: 90 Accuracy: 0.1468
Rotation: 105 Accuracy: 0.21
Rotation: 120 Accuracy: 0.2807
Rotation: 135 Accuracy: 0.3534
Rotation: 150 Accuracy: 0.4108
Rotation: 165 Accuracy: 0.4376
```

#### CapsNet with two Digit Layers

Weights are available [here]().

```
```

#### Resnet 110

Weights are available [here](https://drive.google.com/open?id=1N_x-_5biKY0MaVLQvM71nmlruXdSn6Wy).

```
Rotation: -180 Accuracy: 0.4107
Rotation: -165 Accuracy: 0.4202
Rotation: -150 Accuracy: 0.4109
Rotation: -135 Accuracy: 0.3589
Rotation: -120 Accuracy: 0.2669
Rotation: -105 Accuracy: 0.1897
Rotation: -90 Accuracy: 0.1697
Rotation: -75 Accuracy: 0.2291
Rotation: -60 Accuracy: 0.4264
Rotation: -45 Accuracy: 0.7224
Rotation: -30 Accuracy: 0.931
Rotation: -15 Accuracy: 0.989
Rotation: 0 Accuracy: 0.9959
Rotation: 15 Accuracy: 0.9874
Rotation: 30 Accuracy: 0.8769
Rotation: 45 Accuracy: 0.5646
Rotation: 60 Accuracy: 0.2712
Rotation: 75 Accuracy: 0.1516
Rotation: 90 Accuracy: 0.1531
Rotation: 105 Accuracy: 0.2012
Rotation: 120 Accuracy: 0.2718
Rotation: 135 Accuracy: 0.3484
Rotation: 150 Accuracy: 0.3939
Rotation: 165 Accuracy: 0.4108
```

## Used model implementations

- [Resnet v2](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) by Keras Team

- [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras) by XifengGuo based upon [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)  
