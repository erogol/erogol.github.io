---
layout: post
title: "How does Feature Extraction work on Images?"
description: "Here I share enhanced version of one of my Quora answer to a similar question"
tags: autoencoders computer vision deep learning feature extraction machine learning
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Here I share enhanced version of one of my Quora answer to a similar question ...

There is no single answer for this question since there are many diverse set of methods to extract feature from an image.

First, what is called feature? "a distinctive attribute or aspect of something." so the thing is to have some set of values for a particular instance that diverse that instance from the counterparts. In the field of images, features might be raw pixels for simple problems like digit recognition of well-known [Mnist](http://yann.lecun.com/exdb/mnist/) dataset. However, in natural images, usage of simple image pixels are not descriptive enough. Instead there are two main steam to follow. One is to use hand engineered feature extraction methods (e.g. SIFT, VLAD, HOG, GIST, LBP) and the another stream is to learn features that are discriminative in the given context (i.e. Sparse Coding, Auto Encoders, Restricted Boltzmann Machines, PCA, ICA, K-means). Note that second alternative, representation learning, is the hot wheeled way nowadays.

I will give two examples, one for each stream. One of the prevelant hand engineered method is SIFT ( [Scale-invariant feature transform](http://en.wikipedia.org/wiki/Scale-invariant_feature_transform)) or [Best Explanation I found](http://www.aishack.in/2010/05/sift-scale-invariant-feature-transform/)). From layman's bed, SIFT starts by detecting edges and corners in the image. On the resulted image, SIFT tries to find interesting (a.k.a Region of Interest) points that are differentiating that image from the others. Then, out of each ROI, it extracts a histogram where each of the bins is count of particular edge or corner orientation. These histograms can be concatenated or quantized into some  smaller number of groups with a clustering method like K-means. (I explained SIFT here very naively  since the exact formulation is more complicated and requires some level of Computer Vision and Calculus knowledge). Below, there are some SIFT figures. Hope these help 🙂

Edge Detection

![](http://qph.is.quoracdn.net/main-qimg-c8e4e71c422c34a3cb18ee9323085c10?convert_to_webp=true)

Histogram computation

![](http://qph.is.quoracdn.net/main-qimg-54e13abc636ad8d6f3dbbb30407a8cf7?convert_to_webp=true)

Second stream, representation learning, is mainly described by Deep Learning algorithms or Sparse Coding methods.  Again in the layman term, the ideas is to learn a group of filters that are able to discern one category of images from the another category with some supervised or unsupervised algorithm. For instance, if you use standard Multiple Layer Perceptron for classification, it actually learns a different set of filters  on each layer, in a supervised setting. On the other side, you might prefer to use unsupervised algorithm like [AutoEncoders](http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders). AutoEncoders try to learn latent representations of the given set of instances with lowest possible reconstruction error. In that way,  we are able to learn a supposedly compressed representation of the data in-which each of the latent variable captures a different commonalities incur the images. For example one latent unit might be active if there is an eye on the given image or one another might be active with a noise kind of shape.

General Scheme of Auto Encoders. L1 is the input layer, possibly raw-pixel intensities. L2 is the compressed learned latent representation and L3 is the reconstruction of the given L1 layer from L2 layer. AutoEncoders tries to minimize the difference between L1 and L3 layers with some sparsity constraint.

![](http://qph.is.quoracdn.net/main-qimg-36fc1937240eb6a502670c5ee120d321?convert_to_webp=true)

Example Filters Learned by a Auto Encoder. Each of the filter is receptive to a different edge orientation.

![](http://qph.is.quoracdn.net/main-qimg-7628a7b1e8449354d5695b810d28f4ee?convert_to_webp=true)


### Related posts:

1. [Our ECCV2014 work "ConceptMap: Mining noisy web data for concept learning"](http://www.erogol.com/eccv2014-work-conceptmap-mining-noisy-web-data-concept-learning/ "Our ECCV2014 work \"ConceptMap: Mining noisy web data for concept learning\"")
2. [Intro. to Contractive Auto-Encoders](http://www.erogol.com/intro-contractive-auto-encoders/ "Intro. to Contractive Auto-Encoders")
3. [Recent Advances in Deep Learning](http://www.erogol.com/recent-advances-in-deep-learning/ "Recent Advances in Deep Learning")
4. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")