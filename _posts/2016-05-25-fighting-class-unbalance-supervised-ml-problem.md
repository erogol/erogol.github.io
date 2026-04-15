---
layout: post
title: "Fighting against class imbalance in a supervised ML problem."
description: "ML on imbalanced data

given a imbalanced learning problem with a large class and a small class with"
tags: imbalance machine learning
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

ML on imbalanced data

given a imbalanced learning problem with a large class and a small class with number of instances N and M respectively;

* cluster the larger class into M clusters and use cluster centers for training the model.
* If it is a neural network or some compatible model. Cluster the the large class into K clusters and use these clusters as pseudo classes to train your model. This method is also useful for training your network with small number of classes case. It pushes your net to learn fine-detailed representations.
* Divide large class into subsets with M instances then train multiple classifiers and use the ensemble.
* Hard-mining is a solution which is unfortunately akin to over-fitting but yields good results in some particular cases such as object detection. The idea is to select the most confusing instances from the large set per iteration. Thus, select M most confusing instances from the large class and use for that iteration and repeat for the next iteration.
* For specially batch learning, frequency based batch sampling might be useful. For each batch you can sample instances from the small class by the probability M/(M+N) and N/(M+N) for tha large class so taht you prioritize the small class instances for being the next batch. As you do data augmentation techniques like in CNN models, mostly repeating instances of small class is not a big problem.

Note for metrics, normal accuracy rate is not a good measure for suh problems since you see very high accuracy if your model just predicts the larger class for all the instances. Instead prefer ROC curve or keep watching Precision and Recall.

Please keep me updated if you know something more. Even, this is a very common issue in practice,  still hard to find a working solution.


### Related posts:

1. [Some Basic Machine Learning Terms #1](http://www.erogol.com/some-basic-machine-learning-terms-1/ "Some Basic Machine Learning Terms #1")
2. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
3. [Simple Parallel Processing in Python](http://www.erogol.com/simple-parallel-processing-python/ "Simple Parallel Processing in Python")
4. [Some Useful Machine Learning Libraries.](http://www.erogol.com/broad-view-machine-learning-libraries/ "Some Useful Machine Learning Libraries.")