---
layout: post
title: "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?"
description: "paper: <http://arxiv"
tags: deep learning machine learning paper review
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

paper: <http://arxiv.org/pdf/1511.07543v3.pdf>  
code : <https://github.com/yixuanli/convergent_learning>

This paper is an interesting work which tries to explain similarities and differences between representation learned by different networks in the same architecture.

To the extend of their experiments, they train 4 different AlexNet and compare the units of these networks by correlation and mutual information analysis.

They asks following question;

* Can we find one to one matching of units between network , showing that these units are sensitive to similar or the same commonalities on the image?
* Is the one to one matching stays the same by different similarity measures? They first use correlation then mutual information to confirm the findings.
* Is a representation learned by a network is a rotated version of the other, to the extend that one to one matching is not possible  between networks?
* Is clustering plausible for grouping units in different networks?

Answers to these questions are as follows;

* It is possible to find good matching units with really high correlation values but there are some units learning unique representation that are not replicated by the others. The degree of representational divergence between networks goes higher with the number of layers. Hence, we see large correlations by conv1 layers and it the value decreases toward conv5 and it is minimum by conv4 layer.
* They first analyze layers by the correlation values among units. Then they measure the overlap with the mutual information and the results are confirming each other..
* To see the differences between learned representation, they use a very smart trick. They approximate representations  learned by a layer of a network by the another network using the same layer.  A sparse approximation is performed using LASSO. The result indicating that some units are approximated well with 1 or 2 units of the other network but remaining set of units require almost 4 counterpart units for good approximation. It shows that some units having good one to one matching has local codes learned and other units have slight distributed codes approximated by multiple counterpart units.
* They also run a hierarchical clustering in order to group similar units successfully.

For details please refer to the paper.

**My discussion:**We see that different networks learn similar representations with some level of accompanying uniqueness. It is intriguing  to see that, after this paper, these  are the unique representations causing performance differences between networks and whether the effect is improving or worsening. Additionally, maybe we might combine these differences at the end to improve network performances by some set of smart tricks.

One deficit of the paper is that they do not experiment deep networks which are the real deal of the time. As we see from the results, as the layers go deeper,  different abstractions exhumed by different networks. I believe this is more harsh by deeper architectures such as Inception or VGG kind.

One another curious thing is to study Residual netwrosk. The intuition of Residual networks to pass the already learned representation to upper layers and adding more to residual channel if something useful learned by the next layer. That idea shows some promise that two residual networks might be more similar compared to two Inception networks. Moreover, we can compare different layers inside a single Residual Network to see at what level the representation stays the same.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
2. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")
3. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
4. [NegOut: Substitute for MaxOut units](http://www.erogol.com/negout-substitute-for-maxout-units-2/ "NegOut: Substitute for MaxOut units")