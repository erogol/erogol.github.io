---
layout: post
title: "Paper Review: Self-Normalizing Neural Networks"
description: "One of the main problems of neural networks is to tame layer activations so that one is able to obta"
tags: batch normalization deep learning elu machine learning paper review
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

One of the main problems of neural networks is to tame layer activations so that one is able to obtain stable gradients to learn faster without any confining factor. [Batch Normalization](https://arxiv.org/abs/1502.03167) shows us that keeping values with mean 0 and variance 1 seems to work things. However, albeit indisputable effectiveness of BN, it adds more layers and computations to your model that you'd not like to have in the best case.

[ELU](https://arxiv.org/pdf/1511.07289.pdf) (Exponential Linear Unit) is a activation function aiming to tame neural networks on the fly by a slight modification of activation function. It keeps the positive values as it is and exponentially skew negative values.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/06/elu.png)

ELU function. ![\alpha](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_7b7f9dbfea05c83784f8b85149852f08.gif) is a constant you define.

ELU does its job good enough, if you like to evade the cost of Bath Normalization, however its effectiveness does not rely on a theoretical proof beside empirical satisfaction. And finding a good ![\alpha](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_7b7f9dbfea05c83784f8b85149852f08.gif) is just a guess.

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)takes things to next level. In short, it describes a new activation function SELU (Scaled Exponential Linear Units), a new initialization scheme and a new dropout variant as a repercussion,

The main topic here is to keep network activation in a certain basin defined by a mean and a variance values. These can be any values of your choice but for the paper it is mean 0 and variance 1 (similar to notion of Batch Normalization). The question afterward is to modifying ELU function by some scaling factors to keep the activations with that mean and variance on the fly. They find these scaling values by a long theoretical justification. Stating that, scaling factors of ELU are supposed to be defined as such any passing value of ELU should be contracted to define mean and variance.  (This is just verbal definition by no means complete. Please refer to paper to be more into theory side. )

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/06/selu.png)

Above, the scaling factors are shown as ![\alpha](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_7b7f9dbfea05c83784f8b85149852f08.gif) and ![\lambda](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_c6a6eb61fd9c6c913da73b3642ca147d.gif).  After long run of computations these values appears to be 1.6732632423543772848170429916717 and 1.0507009873554804934193349852946 relatively. Nevertheless, do not forget that these scaling factors are targeting specifically mean 0 and variance 1.  Any change preludes to change these values as well.

Initialization is also another important part of the whole method. The aim here is to start with the right values. They suggest to sample weights from a Gaussian distribution with mean 0 and variance ![1/n](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_878bd532f1718635c637124be801e4d9.gif) where n is number of weights.

It is known with a well credence that Dropout does not play well with Batch Normalization since it smarting network activations in a purely random manner. This method seems even more brittle to dropout effect. As a cure, they propose Alpha Dropout. It randomly sets inputs to saturatied negative value of SELU which is ![-\alpha\lambda](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_d6e81bb294e2282f740b7614955a7f6f.gif). Then an affine transformation is applied to it with ![a](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_0cc175b9c0f1b6a831c399e269772661.gif) and ![b](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_92eb5ffee6ae2fec3ad71c777531578f.gif) values computed relative to dropout rate, targeted mean and variance.It randomizes network without degrading network properties.

In a practical point of view, SELU seems promising by reducing the computation time relative to RELU+BN for normalizing the network. In the paper they does not provide any vision based baseline such a MNIST, CIFAR and they only pounce on Fully-Connected models. I am still curios to see its performance vis-a-vis on these benchmarks agains Bath Normalization. I plan to give it a shoot in near future.

One tickle in my mind after reading the paper is the obsession of mean 0 and variance 1 for not only this paper but also the other normalization techniques. In deed, these values are just relative so why 0 and 1 but not 0 and 4. If you have a answer to this please ping me below.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
2. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
3. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")
4. [Paper Notes: The Shattered Gradients Problem ...](http://www.erogol.com/paper-notes-shattered-gradients-problem/ "Paper Notes: The Shattered Gradients Problem ...")