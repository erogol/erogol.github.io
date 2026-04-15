---
layout: post
title: "What Are Hot Topics for The Current Era of Machine Learning"
description: "1"
tags: machine learning
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

1. Deep learning [5] seems to be getting the most press right now. It is a form of a Neural Network (with many neurons/layers). Articles are currently being published in the New Yorker [1] and the New York Times[2] on Deep Learning.

2. Combining Support Vector Machines (SVMs) and Stochastic Gradient Decent (SGD) is also interesting. SVMs are really interesting and useful because you can use the kernel trick [10] to transform your data and solve a non-linear problem using a linear model (the SVM). A consequence of this method is the training runtime and memory consumption of the SVM scales with size of the data set. This situation makes it very hard to train SVMs on large data sets. SGD is a method that uses a random process to allow machine learning algorithms to converge faster. To make a long story short, you can combine SVMs and SGD to train SVMs on larger data sets (theoretically). For more info, read this link[4].

3. Because computers are now fast, cheap, and plentiful, Bayesian statistics is now becoming very popular again (this is definitely not "new"). For a long time it was not feasible to use Bayesian techniques because you would need to perform probabilistic integrations by hand (when calculating the evidence). Today, Bayesist are using Monte Carlo Markov Chains[6], Grid Approximations[7], Gibbs Sampling[8], Metropolis Algorithm [13], etc. For more information, watch the videos on Bayesian Networks on Coursera. or a read these books [11], [12] (They're da bomb!!!)

4. Any of the algorithms described in the paper "Map Reduce for Machine Learning on a Multicore"[3]. This paper talks about how to take a machine learning algorithm/problem and distribute it across multiple computers/cores. It has very important implications because it means that all of the algorithms mentioned in the paper can be translated into a map-reduce format and distributed across a cluster of computers. Essentially, there would never be a situation where the data set is too large because you could just add more computers to the Hadoop cluster. This paper was published a while ago, but not all of the algorithms have been implemented into Mahout yet.

Machine learning is a really large field of study. I am sure there are a lot more topics but these are four I definitely find interesting.

[1] [Is “Deep Learning” a Revolution in Artificial Intelligence?](http://www.newyorker.com/online/blogs/newsdesk/2012/11/is-deep-learning-a-revolution-in-artificial-intelligence.html)

[2] [Scientists See Advances in Deep Learning, a Part of Artificial Intelligence](http://www.nytimes.com/2012/11/24/science/scientists-see-advances-in-deep-learning-a-part-of-artificial-intelligence.html?_r=0)

[3] [http://www.cs.stanford.edu/peopl...](http://www.cs.stanford.edu/people/ang//papers/nips06-mapreducemulticore.pdf)

[4] [Kernel Approximations for Efficient SVMs (and other feature extraction methods) [update]](http://peekaboo-vision.blogspot.com/2012/12/kernel-approximations-for-efficient.html)

[5] [Deep learning](http://en.wikipedia.org/wiki/Deep_learning)

[6] [Markov chain Monte Carlo](http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)

[7] [http://www.people.fas.harvard.ed...](http://www.people.fas.harvard.edu/~plam/teaching/methods/grid/grid_print.pdf)

[8] [Gibbs sampling](http://en.wikipedia.org/wiki/Gibbs_sampling)

[9] [Coursera](https://www.coursera.org/course/pgm)

[10] [Kernel trick](http://en.wikipedia.org/wiki/Kernel_trick)

[11] [Doing Bayesian Data Analysis](http://www.indiana.edu/~kruschke/DoingBayesianDataAnalysis/)

[12] [Amazon.com: Probability Theory: The Logic of Science (9780521592710): E. T. Jaynes, G. Larry Bretthorst: Books](http://www.amazon.com/Probability-Theory-Science-T-Jaynes/dp/0521592712)

[13] [Metropolis–Hastings algorithm](http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)


### Related posts:

1. [What is Bayes' Theorem?](http://www.erogol.com/what-is-bayes-theorem/ "What is Bayes' Theorem?")
2. [Some possible Matrix Algebra libraries based on C/C++](http://www.erogol.com/some-possible-matrix-algebra-libraries-based-on-cc/ "Some possible Matrix Algebra libraries based on C/C++")
3. [Kohonen Learning Procedure K-Means vs Lloyd's K-means](http://www.erogol.com/kohonen-learning-k-means/ "Kohonen Learning Procedure K-Means vs Lloyd's K-means")
4. [Gradient Boosted Trees Notes](http://www.erogol.com/gradient-boosted-trees-notes/ "Gradient Boosted Trees Notes")