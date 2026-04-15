---
layout: post
title: "What I read lately"
description: " CATEGORICAL REPARAMETERIZATION WITH GUMBEL SOFTMAX

 Link: <https://arxiv"
tags: deep learning paper review papers read-history research
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

##### CATEGORICAL REPARAMETERIZATION WITH GUMBEL SOFTMAX

* Link: <https://arxiv.org/pdf/1611.01144v1.pdf>
* Continuous distribution on the simplex which approximates discrete vectors (one hot vectors) and differentiable by its parameters with reparametrization trick used in VAE.
* It is used for semi-supervised learning.

##### DEEP UNSUPERVISED LEARNING WITH SPATIAL CONTRASTING

* Learning useful unsupervised image representations by using triplet loss on image patches. The triplet is defined by two image patches from the same images as the anchor and the positive instances and a patch from a different image which is the negative.  It gives a good boost on CIFAR-10 after using it as a pretraning method.
* How would you apply to real and large scale classification problem?

##### UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION

* [my post on this](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/)

##### MULTI-RESIDUAL NETWORKS

* For 110-layers ResNet the most contribution to gradient updates come from the paths with 10-34 layers.
* ResNet trained with only these effective paths has comparable performance with the full ResNet. It is done by sampling paths with lengths in the effective range for each mini-batch.
* Instead of going deeper adding more residual connections provides more boost due to the notion of exponential ensemble of shallow networks by the residual connections.
* Removing a residual block from a ResNet has negligible drop on performance in test time in contrast to VGG and GoogleNet.

[Share](https://www.addtoany.com/share)

### Related posts:

1. [Harnessing Deep Neural Networks with Logic Rules](http://www.erogol.com/harnessing-deep-neural-networks-with-logic-rules/ "Harnessing Deep Neural Networks with Logic Rules")
2. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")