---
layout: post
title: "Comparison: SGD vs Momentum vs RMSprop vs Momentum+RMSprop vs AdaGrad"
description: "In this post I'll briefly introduce some update tricks for training of your ML model"
tags: deep learning machine learning momentum neural network optimization
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

In this post I'll briefly introduce some update tricks for training of your ML model. Then, I will present my empirical findings with a linked [NOTEBOOK](http://nbviewer.ipython.org/gist/erogol/b30ca3da1c77b429e854)that uses 2 layer Neural Network on CIFAR dataset.

I assume at least you know what is Stochastic Gradient Descent (SGD). If you don't, you can follow [this tutorial](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/) .  Beside, I'll consider some improvements of SGD rule that result better performance and faster convergence.

SGD is basically a way of optimizing your model parameters based on the gradient information of your loss function (Means Square Error, Cross-Entropy Error ... ). We can formulate this;

![w(t) = w(t-1) - \epsilon * \bigtriangleup w(t)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_1c1400dfcbc64d483c0739c6b97263fa.gif)w(t) = w(t-1) - \epsilon \* \bigtriangleup w(t)

![w](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_f1290186a5d0b1ceab27f4e77c0c5d68.gif) is the model parameter, ![\epsilon](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_92e4da341fe8f4cd46192f21b6ff3aa7.gif) is learning rate and ![\bigtriangleup w(t)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_de8eb8b3c273561ce7e7a9d45410b5d8.gif) is the gradient at the time ![t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_e358efa489f58062f10dd7316b65649e.gif).

SGD as itself  is solely depending on the given instance (or the batch of instances) of the present iteration. Therefore, it  tends to have unstable update steps per iteration and corollary convergence takes more time or even your model is akin to stuck into a poor local minima.

To solve this problem, we can use Momentum idea (Nesterov Momentum in literature). Intuitively, what momentum does is to keep the history of the previous update steps and combine this information with the next gradient step to keep the resulting updates stable and conforming the optimization history. It basically, prevents chaotic jumps.  We can formulate  Momentum technique as follows;

![v(t) = \alpha v(t-1) - \epsilon \frac{\partial E}{\partial w}(t) ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_9e18a7b7b4717f6d3c010b92d2dbe5d8.gif)  (update velocity history with the new gradient)

![\bigtriangleup w(t) = v(t)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_2558fd17359f0411bb3f82a0acf9a0a1.gif) (The weight change is equal to the current velocity)

![\alpha](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_7b7f9dbfea05c83784f8b85149852f08.gif) is the momentum coefficient and 0.9 is a value to start. ![\frac{\partial E}{\partial w}(t)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_b61c4ca9bb3091835ea0ed071ccd7e2c.gif) is the derivative of ![w](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_f1290186a5d0b1ceab27f4e77c0c5d68.gif) wrt. the loss.

Okay we now soothe wild SGD updates with the moderation of Momentum lookup. But still nature of SGD proposes another potential problem. The idea behind SGD is to approximate the real update step by taking the average of the all given instances (or mini batches). Now think about a case where  a model parameter gets a gradient of +0.001 for each  instances then suddenly it gets -0.009 for a particular instance and this instance is possibly a outlier. Then it destroys all the gradient information before. The solution to such problem is suggested by G. Hinton in the Coursera course lecture 6 and this is an unpublished work even I believe it is worthy of.  This is called RMSprop. It keeps running average of its recent gradient magnitudes and divides the next gradient by this average so that loosely gradient values are normalized. RMSprop is performed as below;

![MeanSquare(w,t) =0.9 MeansSquare(w, t-1)+0.1\frac{\partial E}{\partial w}(t)^2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_547088de5ba2972b38438fd17e59a1fb.gif)

![\bigtriangleup w(t) = \epsilon\frac{\partial E}{\partial w}(t) / (\sqrt{MeanSquare(w,t)} + \mu)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_67968828e6eebefcc250d8c72b0c81a7.gif)

![\mu](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_c9faf6ead2cd2c2187bd943488de1d0a.gif) is a smoothing value for numerical convention.

You can also combine Momentum and RMSprop by applying successively and aggregating their update values.

Lets add AdaGrad before finish. AdaGrad is an Adaptive Gradient Method that implies different adaptive learning rates for each feature. Hence it is more intuitive for especially sparse problems and it is likely to find more discriminative features and filters for your Convolutional NN. Although you provide an initial learning rate, AdaGrad tunes it regarding the history of the gradients for each feature dimension. The formulation of AdaGrad is as below;

![w_i(t) = w_i(t-1) + \frac{\epsilon}{\sum_{k=1}^{t}\sqrt{{g_{ki}}^2}}](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_18f62c40352f11601cb58052c28a8ef3.gif)w\_i(t) = w\_i(t-1) + \frac{\epsilon}{\sum\_{k=1}^{t}\sqrt{{g\_{ki}}^2}}  where ![g_{ki} = \frac{\partial E}{\partial w_i}](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_194b28b738a32963756997d10499fa9b.gif)g\_{ki} = \frac{\partial E}{\partial w\_i}

So the upper formula states that, for each feature dimension, learning rate is divided by the all the squared root gradient history.

Now you completed my intro to the applied ideas in this [NOTEBOOK](http://nbviewer.ipython.org/gist/erogol/b30ca3da1c77b429e854) and you can see the practical results of these applied ideas on CIFAR dataset. Of course this into does not mean complete by itself. If you need more refer to other resources. I really suggest the Coursera NN course by G. Hinton for RMSprop idea and [this notes](http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf) for AdaGrad.

For more information you can look this great [lecture slide](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) from Toronto Group.

Lately, I found this great [visualization of optimization methods](http://www.robertsdionne.com/bouncingball/). I really suggest you to take a look at it.



---

**Related posts:**

1. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
2. [Some possible ways to faster Neural Network Backpropagation Learning #1](http://www.erogol.com/some-possible-ways-to-faster-neural-network-backpropagation-learning-1/ "Some possible ways to faster Neural Network Backpropagation Learning #1")
3. [What is special about rectifier neural units used in NN learning?](http://www.erogol.com/what-is-special-about-rectifier-neural-units-used-in-nn-learning/ "What is special about rectifier neural units used in NN learning?")
5. [Microsot Research introduced a new NN model that beats Google and the others](http://www.erogol.com/microsot-research-introduced-new-nn-model-beats-google-others/ "Microsot Research introduced a new NN model that beats Google and the others")