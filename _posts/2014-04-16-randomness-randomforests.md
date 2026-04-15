---
layout: post
title: "Randomness and RandomForests"
description: "One of the enhancing use case of randomness subjected to machine learning is Random Forests"
tags: classifier decision trees machine learning random forests
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

One of the enhancing use case of randomness subjected to machine learning is Random Forests. If you are familiar with Decision Tree that is used inasmuch as vast amount of data analysis and machine learning problems, Random Forests is simple to grasp.

For the beginners, decision tree is a simple, deterministic data structure for modelling decision rules for a specific classification problem (Theoretically shortest possible message length in Information jargon). At each node, one feature is selected to make instance separating decision. That is, we select the feature that separates instances to classes with the best possible "purity". This "purity" is measured by  entropy, gini index or information gain. As lowing to the leaves , tree is branching to disperse the different class of instance to different root to leaf paths.  Therefore, at the leaves of the tree we are able to classify the items to the classes.

However the problem about the decision tree algorithm, it is so fragile to slight changes of the data since these changes are able to change the tree drastically (Decision trees are kind of deterministic structures). Therefore, the final structure as so the final decisions are bugled. If we consider decision trees on the instance space, it divides the space  into little cornered  regions where each region matching to a root to node path. However, it is limited in some way to divide the space with non linear smooth boundaries per contra to non-linear models such as SVM, or neural network algorithms. That makes Decision Trees poor classifiers.

![](http://qph.is.quoracdn.net/main-qimg-2c00705ba46b23e166c78bbac8815fb0?convert_to_webp=true)

![](http://qph.is.quoracdn.net/main-qimg-48e2ce7f27f9e852d18c41f406d9c052?convert_to_webp=true)

After the Decision Tree algorithm, lets see the improvements on top of Random Forests. The ideas behind Random Forests is to generate multiple little trees from random subsets of data. In that way, each of those small trees gives some group of ill-conditioned (biased) classifiers.  Each of them is capturing different regularities since random subset of the instances are in the interest. At the extreme randomness, it curates nodes from random subset of the features as well. In this way feature based randomness is also used. After you simply create n number of trees in this random way, we are able to obtain more cluttered decision boundaries than the simple lines. (n decision trees are used n a voting scheme to decide about novel instance). In addition you might weight some more decisive trees more relative to others by testing on the validation data. (An example decision surface by Random Forests below)

[![Example decision surface by Random Forests](http://qph.is.quoracdn.net/main-qimg-8c0286f0b1ec02555afd6e625fafcca4?convert_to_webp=true "Example decision surface by Random Forests")](http://qph.is.quoracdn.net/main-qimg-8c0286f0b1ec02555afd6e625fafcca4?convert_to_webp=true)

What are the benefits of randomness at the algorithm;

* Robustness against over-fitting since model is created through dense randomness, its generalization abilities are so better than the other algorithms and with the number of trees you create, accuracy is increasing up to a saturation point.
* Soft thresholding (boundaries) on the instance space.
* Simple but powerful as the non linear classification algorithms.
* Able to run multiple tree in parallel (since no mutual relation between trees contrary to AdaBoost), therefore easy to train with large data.

For more detailed information refer;

[Random forests - classification description](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)



### Related posts:

1. [What is different between Random Forests and Gradient Boosted Trees?](http://www.erogol.com/different-random-forests-gradient-boosted-trees/ "What is different between Random Forests and Gradient Boosted Trees?")
2. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
3. [Random Forests MATLAB implementation powered by C](http://www.erogol.com/random-forests-matlab-implementation-powered-c/ "Random Forests MATLAB implementation powered by C")
4. [Good write for different data representation methos in Mach. Learning.](http://www.erogol.com/good-write-for-different-data-representation-methos-in-mach-learning/ "Good write for different data representation methos in Mach. Learning.")