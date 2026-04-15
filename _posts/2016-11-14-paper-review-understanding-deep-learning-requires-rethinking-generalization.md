---
layout: post
title: "Paper review - Understanding Deep Learning Requires Rethinking Generalization"
description: "Paper: <https://arxiv"
tags: deep learning machine learning paper review
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Paper: <https://arxiv.org/pdf/1611.03530v1.pdf>

This paper states the following phrase. Traditional machine learning frameworks (VC dimensions, Rademacher complexity etc.) trying to explain how learning occurs are not very explanatory for the success of deep learning models and we need more understanding looking from different perspectives.

They rely on following empirical observations;

* Deep networks are able to learn any kind of train data even with white noise instances with random labels. It entails that neural networks have very good brute-force memorization capacity.
* Explicit regularization techniques - dropout, weight decay, batch norm - improves model generalization but it does not mean that same network give poor generalization performance without any of these. For instance, an inception network trained without ant explicit technique has 80.38% top-5 rate where as the same network achieved 83.6% on ImageNet challange with explicit techniques.
* A 2 layers network with 2n+d parameters can learn the function f with n samples in d dimensions. They provide a proof of this statement on appendix section. From the empirical stand-view, they show the network performances on MNIST and CIFAR-10 datasets with 2 layers Multi Layer Perceptron.

Above observations entails following questions and conflicts;

* Traditional notion of learning suggests stronger regularization as we use more powerful models. However, large enough network model is able to memorize any kind of data even if this data is just a random noise. Also, without any further explicit regularization techniques these models are able to generalize well in natural datasets.  It shows us that, conflicting to general belief, brute-force memorization is still a good learning method yielding reasonable generalization performance in test time.
* Classical approaches are poorly suited to explain the success of neural networks and more investigation is imperative in order to understand what is really going on from theoretical view.
* Generalization power of the networks are not really defined by the explicit techniques, instead implicit factors like learning method or the model architecture seems more effective.
* Explanation of generalization is need to be redefined in order to solve the conflicts depicted above.

**My take** :  These large models are able to learn any function (and large does not mean deep anymore) and if there is any kind of information match between the training data and the test data, they are able to generalize well as well. Maybe it might be an explanation to think this models as an ensemble of many millions of smaller models on which is controlled by the zeroing effect of activation functions.  Thus, it is able to memorize any function due to its size and implicated capacity but it still generalize well due-to this ensembling effect.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
2. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
3. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
4. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")