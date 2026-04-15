---
layout: post
title: "Bits and Missings for NNs"
description: " Adversarial instances and robust models
  + Generative Adversarial Network http://arxiv"
tags: deep learning machine learning research notes
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

* **Adversarial instances** and robust models
  + Generative Adversarial Network http://arxiv.org/abs/1406.2661 -  Train classifier net as oppose to another net creating possible adversarial instances as the training evolves.
  + Apply genetic algorithms per N training iteration of net and create some adversarial instances.
  + Apply fast gradient approach to image pixels to generate intruding images.
  + Goodfellow states that DAE or CAE are not full solutions to this problem. (verify why ? )

* **Blind training of nets**
  + We train huge networks in a very brute force fashion. What I mean is, we are using larger and larger models since we do not know how to learn concise and effective models. Instead we rely on redundancy and expect to have at least some units are receptive to discriminating features.

* **Optimization (as always)**
  + It seems inefficient to me to use back-propagation after all these work in the field. Another interesting fact, all the effort in the research community goes to find some new tricks that ease back-propagation flaws. I thing we should replace back-propagation all together instead of daily fleeting solutions.
  + Still use SGD ? Still ?

* **Sparsity ?**
  + After a year of hot discussion for sparse representations and it is similarity to human brain activity, it seems like it's been shelved. I still believe, sparsity is very important part of good data representations. It should be integrated to state of art learning models, not only AutoEncoders.

**DISCLAIMER**:  If you are reading this, this is only captain's note and intended to my own research make up.  So many missing references and novice arguments.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.