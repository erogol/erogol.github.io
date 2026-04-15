---
layout: post
title: "What I read for deep-learning"
description: "Today, I spent some time on two new papers proposing a new way of training very deep neural networks"
tags: deep learning paper review
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Today, I spent some time on two new papers proposing a new way of training very deep neural networks ([Highway-Networks](http://arxiv.org/pdf/1505.00387v1.pdf)) and a new activation function for Auto-Encoders ([ZERO-BIAS AUTOENCODERS AND THE BENEFITS OF](http://arxiv.org/pdf/1402.3337v5.pdf)  
 [CO-ADAPTING FEATURES](http://arxiv.org/pdf/1402.3337v5.pdf)) which evades the use of any regularization methods such as Contraction or Denoising.

Lets start with the first one. [Highway-Networks](http://arxiv.org/pdf/1505.00387v1.pdf) proposes a new activation type similar to LTSM networks and they claim that this peculiar activation is robust to any choice of initialization scheme and learning problems occurred at very deep NNs. It is also incentive to see that they trained models with >100 number of layers. The basic intuition here is to learn a gating function attached to a real activation function that decides to pass the activation or the input itself. Here is the formulation

[![Screenshot from 2015-05-11 11:35:12](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_436,h_64/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-113512.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-113512.png)

[![Screenshot from 2015-05-11 11:36:12](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_350,h_93/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-113612.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-113612.png)

![T(x,W_t )](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_e19bced88ef1948f81b9d24ddbbd7668.gif)T(x,W\_t ) is the gating function and ![H(x,W_H)](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_27a4d2a810b91e1f9e9d38bb2911b962.gif)H(x,W\_H) is the real activation. They use Sigmoid activation for gating and Rectifier for the normal activation in the paper. I also implemented it with Lasagne and tried to replicate the results (I aim to release the code later). It is really impressive to see its ability to learn for 50 layers (this is the most I can for my PC).

The other paper [ZERO-BIAS AUTOENCODERS AND THE BENEFITS OF](http://arxiv.org/pdf/1402.3337v5.pdf) [CO-ADAPTING FEATURES](http://arxiv.org/pdf/1402.3337v5.pdf) suggests the use of non-biased rectifier units for the inference of AEs. You can train your model with a biased Rectifier Unit but at the inference time (test time), you should extract features by ignoring bias term. They show that doing so gives better recognition at CIFAR dataset. They also device a new activation function which has the similar intuition to Highway Networks.  Again, there is a gating unit which thresholds the normal activation function.

[![Screenshot from 2015-05-11 11:44:42](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_214,h_39/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-114442.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-114442.png)

[![Screenshot from 2015-05-11 11:47:27](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_293,h_53/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-114727.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/05/Screenshot-from-2015-05-11-114727.png)

The first equation is the threshold function with a predefined threshold (they use 1 for their experiments).  The second equation shows the reconstruction of the proposed model. Pay attention that, in this equation they use square of a linear activation for thresholding and they call this model TLin  but they also use normal linear function which is called TRec. What this activation does here is to diminish the small activations so that the model is implicitly regularized without any additional regularizer. This is actually good for learning over-complete representation for the given data.

For more than this silly into, please refer to papers 🙂 and warn me for any mistake.

These two papers shows a new coming trend to Deep Learning community which is using complex activation functions . We can call it controlling each unit behavior in a smart way instead of letting them fire naively. My notion also agrees with this idea. I believe even more complication we need for smart units in our deep models like Spike and Slap networks.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.