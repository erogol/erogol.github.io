---
layout: post
title: "NegOut: Substitute for MaxOut units"
description: "Maxout 1 units are well-known and frequently used tools for Deep Neural Networks"
tags: deep learning machine learning maxout negout
minute: 5
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Maxout [1] units are well-known and frequently used tools for Deep Neural Networks. For whom does not know, with a basic explanation, a Maxout unit is a set of internal activation units competing with each other for each instance and activation of the winner is propagated as output and the loosers are kept silent. At the backpropagation phase, it means we update only the winner unit. That also means, implicitly, we always prefer to back-propagate gradient signal through the strongest path.  It is an important aspect of Maxout units, especially for very deep models which are prone to gradient instability.

Although Maxout units have very good properties like which I told (please refer to the paper for more details), I am a proactive sceptic of its ability to encode underlying information and pass it to next layer.  Here is a very simple example. Suppose we have two competing functions (filters) in a Maxout unit. One of these functions is receptive of edge structures whereas the other is receptive of corners. For an instance, we might have the first filter as the winner with a value, let’s say, ~3 which means Maxout output is also ~3. For another instance, we have the other function as the winner with approximately same value ~3. If we assume that each NN layer is a classifier which takes the previous layer output as a feature vector (I guess not very wrong assumption), then basically we give the same value for different detections for a particular feature dimension (which is corresponded to our Maxout unit). Eventually, we cannot expect from the next layer to be able to discern this signal.

[![Edge detector is the winner](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Maxout.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Maxout.png)

Edge detector is the winner

[![Corner detector is the winner but the result is same](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Maxout2.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Maxout2.png)

Corner detector is the winner but the result is same

One can argue that we should evaluate Maxout unit as a whole and it is reminiscent of OR function on top of multiple filters. This is a valid argument which I cannot refuse directly but the problem that I indicated above is still floating on air.  Beside,  why we would waste our expensive NN parameters, if we could come up with a better encoding scheme for Maxout units

Here is one alternative approach for better encoding of competing functions, which we call NegOut. Let's assume we have a ordering of two competing functions by heart as 1st and 2nd. If the winner is the 1st function, NegOut outputs the 1st function's value and otherwise it outputs the 2nd function but by taking its negative. NegOut yields two assumptions. The first, competing functions are always positive (like ReLU functions ). The second, we have 2 competing functions.

[![NegOut activation with different winners. ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Negout.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/Negout.png)

NegOut activation with different winners.

If we consider the backpropagation signal, the only difference from Maxout unit is to take negative of the gradient signal for the 2nd competing unit, if it is the winner.

As you can see from the figure, the inherent property here is to output different values for different winner detectors in which the value captures both the structural difference and the strength of the winner activation.

I performed some experiments on CIFAR-10 and MNIST comparing Maxout Network with NegOut Network with exact same architectures explained in the Maxout Paper [1].  The table below summarizes results that I observe by the initial runs without any finetunning or hyper-parameter optimization yet. More comparisons on larger datasets are still in progress.

[![Results on CIFAR-10 and MNIST. ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/NegOutTable.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/10/NegOutTable.png)

Results on CIFAR-10 and MNIST after average of 5 different runs.

NegOut give better results on CIFAR, although it is slightly lower on MNIST. Again notice that no tunning has been took a place for our NegOut network where as Maout Network is optimized as described in the paper [1].  In addition, NegOut network uses 2 competing set of units (as it is constrained by its nature) for the last FC layer in comparison to Maxout net which uses 5 competing units. My expectation is to have more difference as we go through larger models and datasets since as we scale up, representational power takes more place for better results.

Here, I tried to give a basic sketch of my recent work by no means complete. Different observations and experiments are still running. I also need to include LWTA [2] for being more fair and grasp more wider aspect of competing units. Please feel free to share your thoughts as well. Any contribution is appreciated.

PS: Lately, I devote myself to analyze the internal dynamics of Neural Networks with different architectures, layers and activation functions. The aim is checking under the hood and analysing any intuitionally well-functioning ideas applied to  Deep Neural Networks. I also expect to share more of my findings at my blog.

[1] Maxout networks IJ Goodfellow, D Warde-Farley, M Mirza, A Courville, Y Bengio arXiv preprint arXiv:1302.4389

[2] Understanding Locally Competitive Networks Rupesh Kumar Srivastava, Jonathan Masci, Faustino Gomez, Jürgen Schmidhuber. http://arxiv.org/abs/1410.1165

[Share](https://www.addtoany.com/share)

### Related posts:

1. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
2. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
3. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
4. [Why do we need better word representations ?](http://www.erogol.com/need-better-word-representations/ "Why do we need better word representations ?")