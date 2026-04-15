---
layout: post
title: "Paper Notes: Intriguing Properties of Neural Networks"
description: "Paper: https://arxiv"
tags: adversarial ai deep learning paper review
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Paper**: https://arxiv.org/abs/1312.6199

This paper studies description of semantic information with higher level units of an network and blind spot of the network models againt adversarial instances. They illustrate the learned semantics inferring maximally activating instances per unit. They also interpret the effect of adversarial examples and their generalization on different network architectures and datasets.

Findings might be summarized as follows;

1. Certain dimensions of the each layer reflects different semantics of data. (This is a well-known fact to this date therefore I skip this to discuss more)
2. Adversarial instances are general to different models and datasets.
3. Adversarial instances are more significant to higher layers of the networks.
4. Auto-Encoders are more resilient to adversarial instances.

#### Adversarial instances are general to different models and datasets.

They posit that advertorials exploiting a particular network architectures are also hard to classify for the others. They illustrate it by creating adversarial instances yielding 100% error-rate on the target network architecture and using these on the another network. It is shown that these adversarial instances are still hard for the other network ( a network with 2% error-rate degraded to 5%). Of course the influence is not that strong compared to the target architecture (which has 100% error-rate).

#### Adversarial instances are more significant to higher layers of networks.

As you go to higher layers of the network, instability induced by adversarial instances increases as they measure by Lipschitz constant. This is justifiable observation with that the higher layers capture more abstract semantics and therefore any perturbation on an input might override the constituted semantic. (For instance a concept of "dog head" might be perturbed to something random).

#### Auto-Encoders are more resilient to adversarial instances.

AE is an unsupervised algorithm and it is different from the other models used in the paper since it learns the implicit distribution of the training data instead of mere discriminant features. Thus, it is expected to be more tolerant to adversarial instances. It is understood by Table2 that AE model needs stronger perturbations to achieve 100% classification error with generated adversarials.

#### My Notes

One intriguing observation is that shallow model with no hidden unit is yet to be more robust to adversarial instance created from the deeper models. It questions the claim of generalization of adversarial instances. I believe, if the term generality is supposed to be hold, then a higher degree of susceptibility ought to be obtained in this example (and in other too).

I also happy to see that unsupervised method is more robust to adversarial as expected since I believe the notion of general AI is only possible with the unsupervised learning which learns the space of data instead of memorizing things. This is also what I plan to examine after this paper to see how the new tools like Variational Auto Encoders behave againt adversarial instance.

I believe that it is really hard to fight with adversarial instances especially, the ones created by counter optimization against a particular supervised model. A supervised model always has flaws to be exploited in this manner since it memorizes things [ref] and when you go beyond its scope (especially with adversarial instances are of low probability), it makes natural mistakes. Beside, it is known that a neural network converges to local minimum due to its non-convex nature. Therefore, by definition, it has such weaknesses.

Adversarial instances are, in practical sense, not a big deal right now.However, this is akin to be a far more important topic, as we journey through a more advanced AI. Right now, a ML model only makes tolerable mistakes. However, consider advanced systems waiting us in a close future with a use of great importance such as deciding who is guilty, who has cancer. Then this is question of far more important means.

[Share](https://www.addtoany.com/share)

### Related posts:

1. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")
2. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
3. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
4. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")