---
layout: post
title: "Paper review: ALL YOU NEED IS A GOOD INIT"
description: "paper: <http://arxiv"
tags: deep learning iclr 2016 initialization machine learning paper review
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

paper: <http://arxiv.org/abs/1511.06422>  
code: <https://github.com/yobibyte/yobiblog/blob/master/posts/all-you-need-is-a-good-init.md>

This work proposes yet another way to initialize your network, namely LUV (Layer-sequential Unit-variance) targeting especially deep networks.  The idea relies on lately served Orthogonal initialization and fine-tuning the weights by the data to have variance of 1 for each layer output.

The scheme follows three stages;

1. Initialize weights by unit variance Gaussian
2. Find components of these weights using SVD
3. Replace the weights with these components
4. By using minibatches of data, try to rescale weights to have variance of 1 for each layer. This iterative procedure is described as below pseudo code.

[![FROM the paper. Pseudo code of the initialization scheme.](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/06/LUV.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/06/LUV.png)

FROM the paper. Pseudo code of the initialization scheme.

In order to describe the code in words, for each iteration we give a new mini-batch and compute the output variance. We compare the computed variance by the threshold we defined as ![Tol_{var}](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_be0558310947b2285e8a1c455609cb49.gif)Tol\_{var} to the target variance 1.   If number of iterations is below the maximum number iterations or the difference is above ![Tol_{var}](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_be0558310947b2285e8a1c455609cb49.gif)Tol\_{var} we rescale the layer weights by the squared variance of the minibatch.  After initializing this layer go on to the next layer.

In essence, what this method does. First, we start with a normal Gaussian initialization which we know that it is not enough for deep networks. Orthogonalization stage, decorrelates the weights so that each unit of the layer starts to learn from particularly different point in the space. At the final stage, LUV iterations rescale the weights and keep the back and forth propagated signals close to a useful variance against vanishing or exploding gradient problem , similar to Batch Normalization but without computational load.  Nevertheless, as also they points, LUV is not interchangeable with BN for especially large datasets like ImageNet. Still, I'd like to see a comparison with LUV vs BN but it is not done or not written to paper (Edit by the Author: Figure 3 on the paper has CIFAR comparison of BN and LUV and ImageNet results are posted on <https://github.com/ducha-aiki/caffenet-benchmark>).

The good side of this method is it works, for at least for my experiments made on ImageNet with different architectures. It is also not too much hurdle to code, if you already have Orthogonal initialization on the hand. Even, if you don't have it, you can start with a Gaussian initialization scheme and skip Orthogonalization stage and directly use LUV iterations. It still works with slight decrease of performance.


### Related posts:

1. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
2. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
3. [Neural Network Loss and Activation Derivatives](http://www.erogol.com/neural-network-loss-and-activation-derivatives/ "Neural Network Loss and Activation Derivatives")
4. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")