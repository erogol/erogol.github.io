---
layout: post
title: "Gradient Boosted Trees Notes"
description: "Gradient Boosted Trees (GBT) is an ensemble mechanism which learns incrementally new trees optimizin"
tags: ensemble gradient boosted trees hyperparameter selection kaggle kaggle notes
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Gradient Boosted Trees (GBT) is an ensemble mechanism which learns incrementally new trees optimizing the present ensemble's residual error.  This residual error is resemblance to a gradient step of a linear model. A GBT tries to estimate gradient steps by a new tree and update the present ensemble with this new tree so that whole model is updated in the optimizing direction. This is not very formal explanation but it gives my intuition.

One formal way to think about GBT is, there are all possible tree constructions and our algorithms is just selects the useful ones for the given data.  Hence, compared to all possible trees,  number of tress constructed in the model is very small. This is similar to constructing all these infinite  number of trees and averaging them with the weights estimated by  LASSO.

GBT includes different hyper parameters mostly for regularization.

* Early Stopping : How many rounds your GBT continue.
* Shrinkage : Limit the update of each tree with the coefficient ![0 < alpha < 1 ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_d3490850430e3384aceef50462f31560.gif)0 < alpha < 1
* Data subsampling: Do not use whole the data for each tree, instead sample instances. In general sample ration ![ n = 0.5 ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_1d57130d3b662f53b7f69259b341e1f6.gif) but it can be lower for larger datasets.
* One side note: Subsampling without shrinkage performs poorly.

Then my initial setting is:

* Run pretty long with many many round observing a validation data loss.
* Use small shrinkage value ![alpha = 0.001](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_21b892c92a47b6d0c527f03a2414105f.gif)
* Sample 0.5 of the data
* Sample 0.9 of the features as well or do the reverse.



### Related posts:

1. [Kaggle Plankton Challenge Winner's Approach](http://www.erogol.com/kaggle-plankton-challenge-winners-approach/ "Kaggle Plankton Challenge Winner's Approach")
2. [What is Bayes' Theorem?](http://www.erogol.com/what-is-bayes-theorem/ "What is Bayes' Theorem?")
3. [Some possible Matrix Algebra libraries based on C/C++](http://www.erogol.com/some-possible-matrix-algebra-libraries-based-on-cc/ "Some possible Matrix Algebra libraries based on C/C++")
4. ["No free lunch" theorem lies about Random Forests](http://www.erogol.com/free-lunch-theorem-lies-random-forests/ "\"No free lunch\" theorem lies about Random Forests")