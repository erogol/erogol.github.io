---
layout: post
title: "Harnessing Deep Neural Networks with Logic Rules"
description: "paper: <http://arxiv"
tags: deep learning paper review
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

paper: <http://arxiv.org/pdf/1603.06318v1.pdf>

This work posits a way to integrate first order logic rules with neural networks structures. It enables to cooperate expert knowledge with the workhorse deep neural networks. For being more specific, given a sentiment analysis problem, you know that if there is "but" in the sentence the sentiment content changes direction along the sentence. Such rules are harnessed with the network.

The method combines two precursor ideas of information distilling [Hinton et al. 2015] and posterior regularization [Ganchev et al. 2010].  We have teacher and student networks. They learn simultaneously.  Student networks directly uses the labelled data and learns model distribution P then given the logic rules, teacher networks adapts distribution Q as keeping it close to P but in the constraints of the given logic rules. That projects what is inside P to distribution Q bounded by the logic rules. as the below figure suggests.

[![harnessfol](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol.png)I don't like to go into deep math since my main purpose is to give the intuition rather than the formulation. However, formulation follows mathematical formulation of first order logic rules suitable to be in a loss function. Then the student loss is defined by the real network loss (cross-entropy) and the loss of the logic rules with a importance weight.

[![harnessfol_form1](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol_form1.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol_form1.png)![theta](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_61a74be60d291cc4678ab46cc1cdaf91.gif) is the student model weight, the first part of the loss is the network loss and the second part is the logic loss. This function distills the information adapted by the given rules into student network.

Teacher network exploits KL divergence to approximate best Q which is close to P with a slack variable.

[![harnessfol_form2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol_form2.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessfol_form2.png)Since the problem is convex, solution van be found by its dual form with closed form solution as below.

[![harnessingfol_form3](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessingfol_form3.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessingfol_form3.png)

So the whole algorithm is as follows;

[![harnessingfol_algo](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessingfol_algo.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/03/harnessingfol_algo.png)For the experiments and use cases of this algorithm please refer to the paper. They show promising results at sentiment classification with convolution networks by definition of such BUT rules to the network.

My take away is, it is perfectly possible to use expert knowledge with the wild deep networks. I guess the recent trend of deep learning shows the same promise. It seems like our wild networks goes to be a efficient learning and inference rule for large graphical probabilistic models with variational methods and such rules imposing methods.  Still such expert knowledge is tenuous in the domain of image recognition problems.

Disclaimer; it is written hastily without any review therefore it is far from being complete but it targets the intuition of the work to make it memorable for latter use.


### Related posts:

1. [XNOR-Net](http://www.erogol.com/xnor-net/ "XNOR-Net")
2. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")