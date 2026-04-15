---
layout: post
title: "Online Hard Example Mining on PyTorch"
description: "Online Hard Example Mining (OHEM) is a way to pick hard examples with reduced computation cost to im"
tags: codebook deep learning hard mining machine learning ohem
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Online Hard Example Mining (OHEM) is a way to pick hard examples with reduced computation cost to improve your network performance on borderline cases which generalize to the general performance. It is mostly used for Object Detection. Suppose you like to train a car detector and you have positive (with car) and negative images (with no car). Now you like to train your network. In practice, you find yourself in many negatives as oppose to relatively much small positives. To this end, it is clever to pick a subset of negatives that are the most informative for your network. Hard Example Mining is the way to go to this.

![](https://www.cc.gatech.edu/~hays/compvision2016/results/proj5/html/akalia6/data/fighard.jpg)

In a detection problem, hard examples corresponds to false positive detection depicted here with red.

In general, to pick a subset of negatives, first you train your network for couple of iterations, then you run your network all along your negative instances then you pick the ones with the greater loss values. However, it is very computationally toilsome since you have possibly millions of images to process, and sub-optimal for your optimization since you freeze your network while picking your hard instances that are not all being used for the next couple of iterations. That is, you assume here all hard negatives you pick are useful for all the next iterations until the next selection. Which is an imperfect assumption especially for large datasets.

Okay, what Online means in this regard. OHEM solves these two aforementioned problems by performing hard example selection batch-wise. Given a batch sized K, it performs regular forward propagation and computes per instance losses. Then, it finds M<K hard examples in the batch with high loss values and it only back-propagates the loss computed over the  selected instances. Smart hah ? 🙂

It reduces computation by running hand to hand with your regular optimization cycle. It also unties the assumption of the foreseen usefulness by picking hard examples per iteration so thus we now really pick the hard examples for each iteration.

If you like to test yourself, here is PyTorch OHEM implemetation that I offer you to use a bit of grain of salt.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [What is special about rectifier neural units used in NN learning?](http://www.erogol.com/what-is-special-about-rectifier-neural-units-used-in-nn-learning/ "What is special about rectifier neural units used in NN learning?")
2. [What is good about Sparse Data Representation in ML?](http://www.erogol.com/goood-sparse-data-representation-ml/ "What is good about Sparse Data Representation in ML?")
3. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
4. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")