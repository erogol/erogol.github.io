---
layout: post
title: "ParseNet: Looking Wider to See Better"
description: "paper:  <http://arxiv"
tags: context deep learning paper review parsenet segmentation
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**paper**:  <http://arxiv.org/pdf/1506.04579v2.pdf>

**code** :  <https://gist.github.com/shelhamer/80667189b218ad570e82>

In this work, they propose two related problems and comes with a simple but functional solution to this. the problems are;

1. Learning object location on the image with Proposal + Classification approach is very tiresome since it needs to classify >1000 patched per image. Therefore, use of end to end pixel-wise segmentation is a better solution as proposed by FCN (Long et al. 2014).
2. FCN oversees the contextual information since it predicts the objects of each pixel independently. Therefore, even the thing on the image is Cat, there might be unrelated predictions for different pixels. They solve this by applying Conditional Random Field (CRF) on top of FCN. This is a way to consider context by using pixel relations.  Nevertheless, this is still not a method that is able to learn end-to-end since CRF needs additional learning stage after FCN.

Based on these two problems they provide ParseNet architecture. It declares contextual information by looking each channel feature map and aggregating the activations values.  These aggregations then merged to be appended to final features of the network as depicted below;

[![Figure from the paper. It shows the problem told above and proposed feature aggregation](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/04/ParseNet1.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/04/ParseNet1.png)

Figure from the paper. It shows the problem told above and proposed feature aggregation

Their experiments construes the effectiveness of the additional contextual features.  Yet there are two important points to consider before using these features together. Due to the scale differences of each layer activations, one needs to normalize first per layer then append them together.  They L2 normalize each layer's feature. However, this results very small feature values which also hinder the network to learn in a fast manner.  As a cure to this, they learn scale parameters to each feature as used by the Batch Normalization method so that they first normalize and scale the values with scaling weights learned from the data.

The takeaway from this paper,  for myself, adding intermediate layer features improves the results with a correct normalization framework and as we add more layers, network is more robust to local changes by the context defined by the aggregated features.

They use VGG16 and fine-tune it for their purpose, VGG net does not use Batch Normalization. Therefore, use of Batch Normalization from the start might evades the need of additional scale parameters even maybe the L2 normalization of aggregated features. This is because, Batch Normalization already scales and shifts the feature values into a common norm.

**Note**: this is a hasty used article sorry for any inconvenience or mistake or stupidly written sentences.


### Related posts:

1. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")
2. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")