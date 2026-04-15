---
layout: post
title: "Paper review: EraseReLU"
description: "paper: <https://arxiv"
tags: deep learning paper review relu
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

paper: <https://arxiv.org/pdf/1709.07634.pdf>

ReLU is defined as a way to train an ensemble of exponential number of linear models due to its zeroing effect. Each iteration means a random set of active units hence, combinations of different linear models. They discuss, relying on the given observation, it might be useful to remove non-linearities for some layers and letting them to learn combination of linearities as the whole layer.

Another argument as poised, some representations are hard to approximate by a stack of non-linear layers. as shown by He et al. 2016. To this end, letting linearities for a subset of layers might ameliorate the condition.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/09/erase_relu.png)

The way they apply EraseReLU is removing the last ReLU layer of each "module". "Module" here is defined depending on the model architecture as shown above.

Experiments show that EraseReLU increases the performance of networks and its effect is larger for deeper networks. It is also more resilient to over-fitting for deep networks. The loss curves also show faster convergence for EraseReLU and the difference more obvious for larger datasets.

**My 2 cents:** Results are not that different on ImageNet but still better to the favor of EraseReLU. Then it might be the case of lucky shoot since there is no confidence interval or variance given for the trainings.

Faster convergence makes sense with the help of second guessing after the paper. Since there are more active units possible it entails to propagate more gradients. **However, all such comments assumes that error signals are always positive. Which is very unlikely. Therefore, more open valves might cause more chaotic back-propagation signal.**

Yet it is very simple idea, it shows faster convergence, better results and a good investgation of ReLU function. It think it is useful and can take its position in my next training session.

**Disclaimer**: This is written hastily in 10 mins. If you think something wrong or even worse let me know :).

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")
2. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")