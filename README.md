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

## Research results

In our work we decided to check CapsNets ability to understand image rotations. We used CIFAR10 and MINIST datasets in series of experiments, training ResNet 110 architecture as classical CNN model representative to compare it with two CapsNet approaches (single/souble caps layers). Models were trained using two kinds of training datasets: unrotated and rotated CIFAR/MNIST.

### Results on CIFAR10

#### Rotated training dataset

Weights are available here:
- [CapsNet with Digit Layer](https://drive.google.com/open?id=1D4dwYZQqVi1nEDJ4wH6SqpNFEcqdXFUG).
- [CapsNet with two Digit Layers](https://drive.google.com/open?id=1ObaKItC4Zn2weHoDGaUyzYSRMNkehepG).
- [Resnet 110](https://drive.google.com/open?id=1Od3W6AWRANFjnbIndykCHyhFgnETjFP5).

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:----:|:----:|
|-180&deg;|__0.4162__|0.3316|0.337 |
|-165&deg;|__0.4185__|0.3184|0.3283|
|-150&deg;|__0.4171__|0.3115|0.323 |
|-135&deg;|__0.3926__|0.29  |0.2941|
|-120&deg;|__0.3847__|0.2629|0.299 |
|-105&deg;|__0.3929__|0.2661|0.3152|
|-90&deg; |__0.4284__|0.2873|0.3315|
|-75&deg; |__0.4992__|0.3111|0.3703|
|-60&deg; |__0.6246__|0.3774|0.457 |
|-45&deg; |__0.7697__|0.5106|0.6071|
|-30&deg; |__0.8791__|0.6544|0.7463|
|-15&deg; |__0.9145__|0.726 |0.8006|
|0&deg;   |__0.9076__|0.7358|0.791 |
|15&deg;  |__0.91__  |0.7192|0.7965|
|30&deg;  |__0.8829__|0.659 |0.7504|
|45&deg;  |__0.7757__|0.5224|0.6058|
|60&deg;  |__0.6423__|0.3749|0.4494|
|75&deg;  |__0.5137__|0.3031|0.3682|
|90&deg;  |__0.4459__|0.2793|0.3285|
|105&deg; |__0.4049__|0.2569|0.3053|
|120&deg; |__0.3834__|0.2545|0.298 |
|135&deg; |__0.384__ |0.2651|0.2968|
|150&deg; |__0.4127__|0.2978|0.3244|
|165&deg; |__0.4217__|0.3134|0.3346|

---
In relation to the 0&deg; accuracy score:

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:----:|:--------:|
|-180&deg;|__0.4586__|0.4507|0.4260    |
|-165&deg;|__0.4611__|0.4327|0.4150    |
|-150&deg;|__0.4596__|0.4233|0.4083    |
|-135&deg;|__0.4326__|0.3941|0.3718    |
|-120&deg;|__0.4239__|0.3573|0.3780    |
|-105&deg;|__0.4329__|0.3616|0.3985    |
|-90&deg; |__0.472__ |0.3905|0.4191    |
|-75&deg; |__0.55__  |0.4228|0.4681    |
|-60&deg; |__0.6882__|0.5129|0.5778    |
|-45&deg; |__0.8480__|0.6939|0.7675    |
|-30&deg; |__0.9686__|0.8894|0.9435    |
|-15&deg; |1.008     |0.9867|__1.0121__|
|0&deg;   |1         |1     |1         |
|15&deg;  |1.003     |0.9774|__1.007__ |
|30&deg;  |__0.9728__|0.8956|0.9487    |
|45&deg;  |__0.8547__|0.71  |0.7659    |
|60&deg;  |__0.7077__|0.5095|0.5681    |
|75&deg;  |__0.566__ |0.4119|0.4655    |
|90&deg;  |__0.4913__|0.3796|0.4153    |
|105&deg; |__0.4461__|0.3491|0.386     |
|120&deg; |__0.4224__|0.3459|0.3767    |
|135&deg; |__0.4231__|0.3603|0.3752    |
|150&deg; |__0.4547__|0.4047|0.4101    |
|165&deg; |__0.4646__|0.4259|0.4230    |

#### Without rotations in training dataset

Weights are available here:
- [CapsNet with single Digit Layer](https://drive.google.com/open?id=1s4HC3c4ZzwvWb77LPq2J_0vNmiiEMlHW).
- [CapsNet with two Digit Layers](https://drive.google.com/open?id=1WwwMXS9R0rXDHyvWczIq_FZjFxiW7L29).
- [Resnet 110](https://drive.google.com/open?id=1ZP_c75dS66UDamOuN_LKLcBxle0yeNuG).

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:----:|:----:|
|-180&deg;|__0.4162__|0.2729|0.3493|
|-165&deg;|__0.3578__|0.2431|0.3259|
|-150&deg;|__0.2859__|0.22  |0.2809|
|-135&deg;|__0.2569__|0.2042|0.2537|
|-120&deg;|__0.2769__|0.1959|0.2461|
|-105&deg;|__0.3288__|0.2006|0.2624|
|-90&deg; |__0.3989__|0.249 |0.304 |
|-75&deg; |__0.3905__|0.2437|0.3043|
|-60&deg; |__0.3864__|0.2559|0.3227|
|-45&deg; |__0.4226__|0.297 |0.3966|
|-30&deg; |__0.6191__|0.4019|0.5314|
|-15&deg; |__0.87__  |0.566 |0.7331|
|0&deg;   |__0.9185__|0.6805|0.8144|
|15&deg;  |__0.8697__|0.5802|0.7431|
|30&deg;  |__0.6332__|0.4143|0.536 |
|45&deg;  |__0.4304__|0.3   |0.4   |
|60&deg;  |__0.3909__|0.2462|0.3309|
|75&deg;  |__0.3863__|0.2345|0.3023|
|90&deg;  |__0.4138__|0.2465|0.3082|
|105&deg; |__0.3345__|0.2083|0.259 |
|120&deg; |__0.2717__|0.1897|0.236 |
|135&deg; |__0.2482__|0.2   |0.2406|
|150&deg; |__0.2858__|0.2146|0.2553|
|165&deg; |__0.3536__|0.2382|0.3046|

---
In relation to the 0&deg; accuracy score:

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:--------:|:--------:|
|-180&deg;|__0.4531__|0.401     |0.4289    |
|-165&deg;|0.3895    |0.3572    |__0.4002__|
|-150&deg;|0.3113    |0.3233    |__0.3449__|
|-135&deg;|0.2797    |0.3001    |__0.3115__|
|-120&deg;|0.3015    |0.2879    |__0.3022__|
|-105&deg;|__0.358__ |0.2948    |0.3222    |
|-90&deg; |__0.4343__|0.3659    |0.3733    |
|-75&deg; |__0.4252__|0.3581    |0.3736    |
|-60&deg; |__0.4207__|0.376     |0.3962    |
|-45&deg; |0.4601    |0.4364    |__0.4870__|
|-30&deg; |__0.6740__|0.5906    |0.6525    |
|-15&deg; |__0.9472__|0.8317    |0.9002    |
|0&deg;   |1         |1         |1         |
|15&deg;  |__0.9469__|0.8526    |0.9125    |
|30&deg;  |__0.6894__|0.6088    |0.6582    |
|45&deg;  |0.4686    |0.4409    |__0.4912__|
|60&deg;  |__0.4256__|0.3618    |0.4063    |
|75&deg;  |__0.4206__|0.3446    |0.3712    |
|90&deg;  |__0.4505__|0.3622    |0.3784    |
|105&deg; |__0.3642__|0.3061    |0.318     |
|120&deg; |__0.2958__|0.2788    |0.2898    |
|135&deg; |0.2702    |0.2939    |__0.2954__|
|150&deg; |0.3112    |__0.3154__|0.3135    |
|165&deg; |__0.385__ |0.3500    |0.374     |

### Results on MNIST

#### Rotated training dataset

Weights are available here:
- [CapsNet with single Digit Layer](https://drive.google.com/open?id=1demuj6N_8PTQWs_vP0hvHFnCSRx8Udx3).
- [CapsNet with two Digit Layers](https://drive.google.com/open?id=1ODsS__IHprAVFhQpOiKCT3u6TqcwfLRG).
- [Resnet 110](https://drive.google.com/open?id=1AUH88BcE8OUE5rg9PdxNlnbXjo7Zh6Z7).

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:--------:|:--------:|
|-180&deg;|0.4216    |0.4261    |__0.4274__|
|-165&deg;|0.4307    |0.4236    |__0.443__ |
|-150&deg;|0.4305    |0.4247    |__0.4343__|
|-135&deg;|__0.4181__|0.4011    |0.4008    |
|-120&deg;|__0.3825__|0.3427    |0.3155    |
|-105&deg;|__0.3649__|0.2936    |0.2448    |
|-90&deg; |__0.4119__|0.3306    |0.2548    |
|-75&deg; |__0.6214__|0.5177    |0.4569    |
|-60&deg; |__0.8523__|0.802     |0.7752    |
|-45&deg; |__0.9673__|0.9582    |0.944     |
|-30&deg; |__0.9901__|__0.9901__|0.9836    |
|-15&deg; |__0.9951__|0.993     |0.9899    |
|0&deg;   |__0.9956__|0.9929    |0.9871    |
|15&deg;  |__0.9961__|0.9919    |0.9913    |
|30&deg;  |__0.9926__|0.989     |0.9883    |
|45&deg;  |__0.9654__|0.9433    |0.9483    |
|60&deg;  |__0.7894__|0.6914    |0.7153    |
|75&deg;  |__0.496__ |0.3606    |0.3844    |
|90&deg;  |__0.337__ |0.2284    |0.235     |
|105&deg; |__0.3524__|0.2705    |0.2653    |
|120&deg; |__0.3983__|0.3785    |0.3577    |
|135&deg; |0.4316    |__0.4427__|0.423     |
|150&deg; |0.4296    |__0.4572__|0.4478    |
|165&deg; |0.4226    |__0.444__ |0.4369    |

#### Without rotations in training dataset

Weights are available here: 
- [CapsNet with single Digit Layer](https://drive.google.com/open?id=1s4KzwmXSCywwWQVQtfZbFd50_9ml4zoJ).
- [CapsNet with two Digit Layers](https://drive.google.com/open?id=1N_x-_5biKY0MaVLQvM71nmlruXdSn6Wy).
- [Resnet 110](https://drive.google.com/open?id=1N_x-_5biKY0MaVLQvM71nmlruXdSn6Wy).

| Rotation degrees | Accuracy - Resnet110 | Accuracy - CapsNet (Single layer) | Accuracy - CapsNet (Double layer) |
|:--------|:--------:|:--------:|:-----:|
|-180&deg;|0.4107    |__0.4408__|0.3968 |
|-165&deg;|0.4202    |__0.4346__|0.405  |
|-150&deg;|__0.4109__|0.4049    |0.3984 |
|-135&deg;|__0.3589__|0.3402    |0.3422 |
|-120&deg;|__0.2669__|0.2597    |0.26   |
|-105&deg;|__0.1897__|0.1881    |0.182  |
|-90&deg; |0.1697    |__0.1733__|0.1637 |
|-75&deg; |0.2291    |__0.2488__|0.2363 |
|-60&deg; |__0.4264__|0.3991    |0.3928 |
|-45&deg; |__0.7224__|0.644     |0.6486 |
|-30&deg; |__0.931__ |0.9004    |0.8928 |
|-15&deg; |__0.989__ |0.9829    |0.9803 |
|0&deg;   |__0.9959__|0.9932    |0.9913 |
|15&deg;  |__0.9874__|0.9815    |0.9726 |
|30&deg;  |__0.8769__|0.8631    |0.8099 |
|45&deg;  |__0.5646__|0.5472    |0.5073 |
|60&deg;  |__0.2712__|0.2637    |0.2289 |
|75&deg;  |__0.1516__|0.147     |0.1222 |
|90&deg;  |__0.1531__|0.1468    |0.134  |
|105&deg; |0.2012    |__0.21__  |0.2024 |
|120&deg; |0.2718    |__0.2807__|0.2806 |
|135&deg; |0.3484    |__0.3534__|0.3336 |
|150&deg; |0.3939    |__0.4108__|0.3734 |
|165&deg; |0.4108    |__0.4376__|0.3965 |

## Used model implementations

- [Resnet v2](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) by Keras Team

- [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras) by XifengGuo based upon [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)  
