---
layout: post
title: "What is different between Random Forests and Gradient Boosted Trees?"
description: "This a simple confusion for especially beginners or the practitioners of Machine Learning"
tags: decision trees machine learning random forests statistical machine learning
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

This a simple confusion for especially beginners or the practitioners of Machine Learning. Therefore, here I share a little space to talk about Random Forests and Gradient Boosted Trees.

To begin with, divide the perspective of differences in to two as algorithmic and practical.

**Algorithmic difference is**; Random Forests are trained with random sample of data (even more randomized cases available like feature randomization) and it trusts randomization to have better generalization performance on out of train set.

On the other spectrum, Gradient Boosted Trees algorithm additionally tries to find optimal linear combination of trees (assume final model is the weighted sum of predictions of individual trees) in relation to given train data. This extra tuning might be deemed as the difference. Note that, there are many variations of those algorithms as well.

**At the practical side**; owing to this tuning stage, Gradient Boosted Trees are more susceptible to jiggling data. This final stage makes GBT more likely to overfit therefore if the test cases are inclined to be so verbose compared to train cases this algorithm starts lacking. On the contrary, Random Forests are better to strain on overfitting although it is lacking on the other way around.

So the best choice depends to the case your have as always.


### Related posts:

1. [Randomness and RandomForests](http://www.erogol.com/randomness-randomforests/ "Randomness and RandomForests")
2. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
3. [Random Forests MATLAB implementation powered by C](http://www.erogol.com/random-forests-matlab-implementation-powered-c/ "Random Forests MATLAB implementation powered by C")
4. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")