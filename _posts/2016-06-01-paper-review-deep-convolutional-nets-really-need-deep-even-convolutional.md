---
layout: post
title: "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?"
description: "There is theoretical proof that any one hidden layer network with enough number of sigmoid function "
tags: deep learning iclr 2016 machine learning paper review
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

There is theoretical proof that any one hidden layer network with enough number of sigmoid function is able to learn any decision boundary. Empirical practice, however, posits us that learning good data representations demands deeper networks, like the last year's ImageNet winner ResNet.

There are two important findings of this work. The first is,we need convolution, for at least image recognition problems, and the second is deeper is always better . Their results are so decisive on even small dataset like CIFAR-10.

They also give a good little paragraph explaining a good way to curate best possible shallow networks based on the deep teachers.

- train state of deep models

- form an ensemble by the best subset

- collect eh predictions on a large enough transfer test

- distill the teacher ensemble knowledge to shallow network.

(if you like to see more about how to apply teacher - student paradigm successfully refer to the paper. It gives very comprehensive set of instructions.)

Still, ass shown by the experimental results also, best possible shallow network is beyond the deep counterpart.

[![FROM PAPER, network performances. As you see with number of layers, performance is also getting better and Teacher is always better then student.](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/06/deepnet.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/06/deepnet.png)

FROM PAPER, network performances. As you see with number of layers, performance is also getting better and Teacher is always better then student.

**My Discussion:**

I believe the success of the deep versus shallow depends not the theoretical basis but the way of practical learning of the networks. If we think networks as representation machine which gives finer details to coerce concepts such as thinking to learn a face without knowing what is an eye, does not seem tangible. Due to the one way information flow of convolution networks, this hierarchy of concepts stays and disables shallow architectures to learn comparable to deep ones.

Then how can we train shallow networks comparable to deep ones, once we have such theoretical justifications. I believe one way is to add intra-layer connections which are connections each unit of one layer to other units of that layer. It might be a recursive connection or just literal connections that gives shallow networks the chance of learning higher abstractions.

Convolution is also obviously necessary. Although, we learn each filter from the whole input, still each filter is receptive to particular local commonalities.  It is not doable by fully connected layers since it learns from the whole spatial range of the input.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")
2. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
3. [NegOut: Substitute for MaxOut units](http://www.erogol.com/negout-substitute-for-maxout-units-2/ "NegOut: Substitute for MaxOut units")
4. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")