---
layout: post
title: "Some possible ways to faster Neural Network Backpropagation Learning #1"
description: "Using Stochastic Gradient instead of Batch Gradient  
Stochastic Gradient:

 faster
 more suitable t"
tags: machine learning neural network
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Using Stochastic Gradient instead of Batch Gradient**  
Stochastic Gradient:

* faster
* more suitable to track changes in each step
* often results with better solution - it may finds different ways to different local minimums on cost function due to it fluctuation on weights -
* Most common way to implement NN learning.

Batch Gradient:

* Analytically more tractable for the way of its convergence
* Many acceleration techniques are suited to Batch L.
* More accurate convergence to local min. - again because of the fluctuation on weights in Stochastic method -

**Shuffling Examples**

* give the more informative instance to algorithm next as the learning step is going further - more informative instance means causing more cost or being unseen -
* Do not give successively instances from same class.

**Transformation of Inputs**

* Mean normalization of input variables around zero mean
* Scale input variables so that covariances are about the same unit length
* Diminish correlations between features as much as possible - since two correlated input may result to learn same function by different units that is redundant -


### Related posts:

1. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
2. [ML WORK-FLOW (Part2) - Data Preprocessing](http://www.erogol.com/ml-work-flow-part2-data-preprocessing/ "ML WORK-FLOW (Part2) - Data Preprocessing")
3. [ML Work-Flow (Part 5) – Feature Preprocessing](http://www.erogol.com/ml-work-flow-part-5-feature-processing/ "ML Work-Flow (Part 5) – Feature Preprocessing")
4. [Microsoft Research introduced a new NN model that beats Google and the others](http://www.erogol.com/microsot-research-introduced-new-nn-model-beats-google-others/ "Microsoft Research introduced a new NN model that beats Google and the others")