---
layout: post
title: "What Are the Needs of Machine Learning?"
description: "–Convexity, including convex optimization and formulation of problems as convex programs"
tags: machine learning
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

–**Convexity**, including convex optimization and formulation of problems as convex programs. Two important subsets of this are linear programming and proximal gradient-style optimization algorithms and formulations, which have a ridiculously vast array of applications for industrial engineering and machine learning.  
–**Probabilistic modeling and inference**: Graphical models and max-entropy models are the most important, and have a vast array of applications in machine learning and more structured statistical modeling. Markov Chain Monte Carlo is a terrific and amazing algorithm with a great special case called Gibbs sampling – they both present almost generic methods of inference on well-specified graphical models. Understand probabilistic formulations both as statistical distributions and objects to optimize.  
–**Regularization**: related to both of the above, in that a lot of regularization has nice convex formulations and probabilistic interpretations (corresponding to Bayesian priors). But there’s  lot more it can do for you – compressed sensing is just L1 regularized least squares (kind of); non-convex models can be effectively regularized, decreasing their VC dimension and making them generalize better.  
–**Competitive analysis**: Basically a way of analyzing algorithms when the task you’re trying to attack has some really large degree of uncertainty and you don’t want to make a bayesian specification of that uncertainty. Instead of asking the acutal guarantees on an algorithm’s performance, ask for it’s guarantee relative to the best algorithm in a class of competing algorithms. E.g. how does my linear classifier that had to predict online do against the best linear classifier in retrospect.  
–**Sparse linear algebra**: I want the first eigenvector of a gazillion X gazillion matrix. How do I get it? This is actually the PageRank problem. Know how to efficiently perform basic linear algebra when dealing with a very sparse matrix (and understand efficiencies from sparsity in lists and from other structural properties too).

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.