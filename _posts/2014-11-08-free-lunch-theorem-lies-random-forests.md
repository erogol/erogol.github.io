---
layout: post
title: "\"No free lunch\" theorem lies about Random Forests"
description: "!download(https://web"
tags: algorithms classifier. comparison machine learning paper_review
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

[![download](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/11/download.gif)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/11/download.gif)

I've read a great paper by Delgado et al.  namely "[Do we Need Hundreds of Classifiers to Solve Real World Classication Problems?](http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf)" in which they compare 179 different classifiers from 17 families on 121 data sets composed by the whole UCI data base and some real-world problems. Classifiers are from R with and without caret pack, C and Matlab (I wish I could see Sklearn as well).

I really recommend you to read the paper in detail but I will share some of the highlights here. The most impressive result is the performance of Random Forests (RF) Implementations. For each dataset, RF is always at the top places. It gets 94.1%  of max accuracy and goes by 90% in the 84.3% of the data sets. Also, 3 out of 5 best classifiers are RF for any data set. This is pretty impressive, I guess. The runner-up is SVM with Gaussian kernel implemented in LibSVM and it archives 92.3% max accuracy. The paper points RF, SVM with Gaussian and Polynomial kernels, Extreme Learning Machines with Gaussian kernel, C5.0 and avNNet (a committe of MLPs implemented in R with caret package) as the top list algorithms after their experiments.

One shortcoming of the paper, from my beloved NN perspective,  is used Neural Network models are not very up-to-date versions such as drop-out, max-out networks. Therefore, it is hard to evaluate algorithms against these advance NN models. However, for anyone in the darn dark of algorithms, it is a quite good guideline that shows the power of RF and SVM against the others.



### Related posts:

1. [Good write for different data representation methos in Mach. Learning.](http://www.erogol.com/good-write-for-different-data-representation-methos-in-mach-learning/ "Good write for different data representation methos in Mach. Learning.")
2. [Kohonen Learning Procedure K-Means vs Lloyd's K-means](http://www.erogol.com/kohonen-learning-k-means/ "Kohonen Learning Procedure K-Means vs Lloyd's K-means")
3. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
4. [A Slide About Model Evaluation Methods](http://www.erogol.com/a-slide-about-model-comparison-methods/ "A Slide About Model Evaluation Methods")