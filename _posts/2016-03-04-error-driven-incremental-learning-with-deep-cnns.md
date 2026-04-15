---
layout: post
title: "Error-Driven Incremental Learning with Deep CNNs"
description: "paper link(http://research"
tags: deeplearning incremental traning paper review traning methods
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

[paper link](http://research.microsoft.com/pubs/238746/error-driven%20incremental%20learning%20in%20deep%20convolutional%20neural%20network%20for%20large-scale%20image%20classification.pdf)

This paper posits a way of incremental training of a network where you have continuous flow of new data categories.  they propose two main problems related to that problem.  First, with increasing number of instances we need more capacitive networks which are hard to train compared to small networks. Therefore starting with a small network and gradually increasing its size seems feasible. Second is to expand the network instead of using already learned features in new tasks. For instance, if you would like to use a pre-trained ImageNet network to your specific problem using it as a sole feature extractor does not reflect the real potential of the network. Instead, training it as it goes wild with the new data is a better choice.

They also recall the forgetting problem when new data is proposed to a pre-trained model. Al ready learned features are forgotten with the new data and the problem.

The proposed method here relies on tree-like structures networks as the below figure depicts. The algorithm starts with a pretrained network L0 with K superclasses. When we add new classes (depicted green), we clone network L0 to leaf networks L1, L2 and branching network B. That is, all set of new networks are the exact clone of L0. Then B is the branching network which decides the leaf network to be activated for the given instance. Then activated leaf network leads to the final prediction for the given instance.

[![incremental1](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/incremental1.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/incremental1.png) For partition the classes the idea is to keep more confusing classes together so that the later stages of the network can resolve this confusion.  So any new set of classes with the corresponding instances are passed through the already trained networks and mostly active network by its softmax outputs is selected for that single category to be appended.  Another choice to increase the number of categories is to add the new categories to output layer only by keeping the network capacity the same. When we need to increase the capacity then we can branch the network again and this network stays as a branching network now. When we need to decide the leaf network following that branching network we sum the confidence values of the classes of each leaf network and maximum confidence network is selected as the leaf network.

[![incremental2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/incremental2.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/incremental2.png)

While all these processes, any parameter is transfered from a branching network to leaf networks unless we have some mismatch between category units. Only these mismatch parameters are initialized randomly.

This work proposes a good approach for a scalable learning architecture with new categories coming in time. It both considers how to add new categories and increase the network capacity in a guided manner. One another good side of this architecture is that each of these network can be trained independently so that we can parallelize the training process.

[Share](https://www.addtoany.com/share)

### Related posts:

1. [What I read for deep-learning](http://www.erogol.com/what-i-read-for-deep-learning/ "What I read for deep-learning")
2. [Great Slide by Alex Smola about MXNet](http://www.erogol.com/great-slide-by-alex-smola-about-mxnet/ "Great Slide by Alex Smola about MXNet")
3. [My Notes - SqueezeNet: AlexNet accuracy with 100X smaller network](http://www.erogol.com/squeezenet-alexnet-accuracy-wit-100x-smaller-network/ "My Notes - SqueezeNet: AlexNet accuracy with 100X smaller network")