---
layout: post
title: "What is special about rectifier neural units used in NN learning?"
description: "Sigmoid unit :  
! f(x) = frac{1}{1+exp(-x)} (http://qlx"
tags: deep learning machine learning neural networ
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Sigmoid unit :**  
![ f(x) = frac{1}{1+exp(-x)} ](http://qlx.is.quoracdn.net/main-371284bd9b148215.png " f(x) = frac{1}{1+exp(-x)} ")  
**Tanh unit:**  
![ f(x) = tanh(x) ](http://qlx.is.quoracdn.net/main-cad75b2bc331520e.png " f(x) = tanh(x) ")

**Rectified linear unit (ReLU):**  
![ f(x) = sum_{i=1}^{inf} sigma (x - i + 0.5) approx log(1 + e^{x}) ](http://qlx.is.quoracdn.net/main-0d297525963fde53.png " f(x) = sum_{i=1}^{inf} sigma (x - i + 0.5) approx log(1 + e^{x}) ")

we refer

* ![ sum_{i=1}^{inf} sigma (x - i + 0.5) ](http://qlx.is.quoracdn.net/main-dd49800d1baae78c.png " sum_{i=1}^{inf} sigma (x - i + 0.5) ") as **stepped sigmoid**

* ![ log(1 + e^{x}) ](http://qlx.is.quoracdn.net/main-d1d27bde5e5b3f74.png " log(1 + e^{x}) ") as **softplus function**

The softplus function can be approximated by **max function** (**or hard max** ) ie   ![ max( 0, x + N(0,1)) ](http://qlx.is.quoracdn.net/main-0bb1581ccfca0415.png " max( 0, x + N(0,1)) ") . The max function is commonly known as **Rectified Linear Function (ReL).**

In the following figure below we compare ReL function (soft/hard) with sigmoid function.

![](http://qph.is.quoracdn.net/main-qimg-cf46ade91ad2015b78270bdff4fd7362)

The major differences between the sigmoid and ReL function are:

* Sigmoid function  has range [0,1] whereas the ReL function has range ![[0,infty] ](http://qlx.is.quoracdn.net/main-38d116470f64b8e9.png "[0,infty] "). Hence sigmoid function can be used to model probability, whereas ReL can be used to model positive real number. NOTE:  The view of softplus function as approximation of stepped sigmoid units relates to the binomial hidden units as discussed in [http://machinelearning.wustl.edu...](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)
* The gradient of the sigmoid function vanishes as we increase or decrease x. However, the gradient of the ReL function doesn't vanish as we increase x. In fact, for max function, gradient is defined as ![ begin{Bmatrix} 0 & if\ x < 0 \\ 1 & if\ x > 0 end{Bmatrix} ](http://qlx.is.quoracdn.net/main-aec6efce214c28d6.png " begin{Bmatrix} 0 & if\ x < 0 \\ 1 & if\ x > 0 end{Bmatrix} ") **.**

The advantages of using Rectified Linear Units in neural networks are

* If hard max function is used as activation function, it  induces the sparsity in the hidden units.
* As discussed earlier ReLU doesn't face gradient vanishing problem as faced by sigmoid and tanh function. Also, It has been shown that deep networks can be trained efficiently using ReLU even without pre-training.
* ReLU can be used in Restricted Boltzmann machine to model real/integer valued inputs.

References :

* On Rectified Linear Units for Speech Processing [http://www.cs.toronto.edu/~hinto...](http://www.cs.toronto.edu/~hinton/absps/googlerectified.pdf)
* Rectifier Nonlinearities Improve Neural Network Acoustic Models [http://ai.stanford.edu/~amaas/pa...](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
* Deep Sparse Rectifier Neural Networks [http://eprints.pascal-network.or...](http://eprints.pascal-network.org/archive/00008596/01/glorot11a.pdf)


### Related posts:

1. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
2. [Some possible ways to faster Neural Network Backpropagation Learning #1](http://www.erogol.com/some-possible-ways-to-faster-neural-network-backpropagation-learning-1/ "Some possible ways to faster Neural Network Backpropagation Learning #1")
3. [What is good about Sparse Data Representation in ML?](http://www.erogol.com/goood-sparse-data-representation-ml/ "What is good about Sparse Data Representation in ML?")
4. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")