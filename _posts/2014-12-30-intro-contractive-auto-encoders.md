---
layout: post
title: "Intro. to Contractive Auto-Encoders"
description: "Contractive Auto-Encoder is a variation of well-known Auto-Encoder algorithm that has a solid backgr"
tags: autoencoders contractive_auto_encoders deep learning machine learning
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Contractive Auto-Encoder is a variation of well-known Auto-Encoder algorithm that has a solid background in the information theory and lately deep learning community. The simple Auto-Encoder targets to compress information of the given data as keeping the reconstruction cost lower as much as possible. However another use is to enlarge the given input's representation. In that case, you learn over-complete representation of the given data instead of compressing it. Most common implication is Sparse Auto-Encoder that learns over-complete representation but in a sparse (smart) manner. That means, for a given instance only informative set of units are activated, therefore you are able to capture more discriminative representation, especially if you use AE for pre-training of your deep neural network.

After this intro. what is special about Contraction Auto-Encoder (CAE)?  CAE simply targets to learn invariant representations to unimportant transformations for the given data. It only learns transformations that are exactly in the given dataset and try to avoid more. For instance, if you have set of car images and they have left and right view points in the dataset, then CAE is sensitive to those changes but it is insensitive to frontal view point. What it means that if you give a frontal car image to CAE after the training phase, it tries to contract its representation to one of the left or right view point car representation at the hidden layer. In that way you obtain some level of view point in-variance. (I know, this is not very good example for a cannier guy but I only try to give some intuition for CAE).

From the mathematical point of view, it gives the effect of contraction by adding an additional term to reconstruction cost. This addition is the Sqrt Frobenius norm of Jacobian of the hidden layer representation with respect to input values. If this value is zero, it means, as we change input values, we don't observe any change on the learned hidden representations. If we get very large values then the learned representation is unstable as the input values change.

This was just a small intro to CAE, if you like the idea please follow the below videos of Hugo Larochelle's lecture and [Pascal Vincent's talk](http://techtalks.tv/talks/contractive-auto-encoders-explicit-invariance-during-feature-extraction/54426/) at ICML 2011 for the paper.

[Share](https://www.addtoany.com/share)

### Related posts:

1. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
2. [What is good about Sparse Data Representation in ML?](http://www.erogol.com/goood-sparse-data-representation-ml/ "What is good about Sparse Data Representation in ML?")
3. [Hinton's NN journey](http://www.erogol.com/hintons-nn-journey/ "Hinton's NN journey")