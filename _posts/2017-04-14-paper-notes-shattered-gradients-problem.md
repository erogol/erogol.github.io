---
layout: post
title: "Paper Notes: The Shattered Gradients Problem ..."
description: "paper: <https://arxiv"
tags: batch normalization deep learning gradient descent optimization paper review
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

paper: <https://arxiv.org/abs/1702.08591>

The whole heading of the paper is "The Shattered Gradients Problem: If resnets are the answer, then what is the question?". It is really interesting work with all its findings about gradient dynamics of neural networks. It also examines Batch Normalization (BN) and Residual Networks (Resnet) under this problem.

The problem, dubbed "Shattered Gradients", described as gradient feedbacks resembling random noise for nearby data points. White noise gradients (random value around 0 with some unknown variance) are not useful for training and they stall the network. What we expect to see is Brownian noise (next value is obtained with a small change on the last value) from a working model. Deep neural networks are more prone to white noise gradients. However, latest advances like BN and Resnet are described to be more resilient to random gradients even in deep networks.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/04/shattered_grads_1.png)

White noise gradients undermines the effectiveness of networks because they violates gradient based learning methods which expects similar gradient feedbacks for data points close by in the vector space. Once you have white noise gradient for such close points, the model is not able to capture data manifold through these learning algorithms. Brownian updates yields more correlation on updates and this preludes effective learning.

For normal networks, they give a empirical evidence that the correlation of network updates decreases with the order ![(/2^L](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_a6f574f819085d8bee5f226890e842a8.gif) where L is number of layers. Decreasing correlation means more white noise gradient feedbacks.

One important reason of white noise feedbacks is to be co-activations of network units. From a working model, we expect to have units receptive to different structures in the given data. Therefore, for each different instance, different subset of units should be active for effective information flux. They observe that as activation goes through layers, co-activation rate goes higher. BN layers prevents this by keeping the co-activation rate 1/4 (1/4 units are active per layer).

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/04/shatter_grad_2.png)

Beside the co-activation rate, how dispersed units activation is another important question. Thus, similar instances need to activate similar subset of units and activation should be distributes to other subsets as we change the data structure. This stage is where the skip-connections get into the play. Their observation is skip-connections improve networks in that respect. This can be observed at below figure.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/04/shatter_grad_3.png)

The effectiveness of skip-connections increases with Beta scaling introduced by InceptionV4 architecture. It  is scaling residual connections by a constant value before summing up with the current layer activation.

#### A small discussion

This is a very intriguing paper to me as being one of the scarse works investigating network dynamics instead of blind updates on architectures for racing accuracy values.

Resnet is known to be train hundreds of layers which was not possible before. Now, with this work, we have another scientific argument explaining its effectiveness. I also like to point Veit et al. (2016) demystifying Resnet as an ensemble of many shallow networks. When we combine both of these papers, it makes total sense to me how Resnets are useful for training very deep networks. If shattered gradient effect, as stated here,  increasing with number of layers with the order 2^L then it is impossible to train hundred layers with an ad-hoc network. Corollary, since Resnet behaves like a ensemble of shallow networks this effects is rehabilitated. We are able to see it empirically in this paper and it is complimentary in that sense.

**Note:** This hastily written paper note might include any kind of error. Please let me know if you find one. Best 🙂

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
2. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
3. [Paper review: Dynamic Capacity Networks](http://www.erogol.com/1314-2/ "Paper review: Dynamic Capacity Networks")
4. [Selfai: A Method for Understanding Beauty in Selfies](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/ "Selfai: A Method for Understanding Beauty in Selfies")