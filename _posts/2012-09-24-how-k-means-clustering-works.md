---
layout: post
title: "How K-means clustering works"
description: "K-means is the most primitive and easy to use clustering algorithm (also a Machine Learning algorith"
tags: algorithm animation gif k-means machine learning
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

K-means is the most primitive and easy to use clustering algorithm (also a Machine Learning algorithm).  
There are 4 basic steps of K-means:

1. Choose K different initial data points on instance space (as initial centroids) - centroid is the mean points of the clusters that overview the attributes of the classes-.
2. Assign each object to the nearest centroid.
3. After all the object are assigned, recalculate the centroids by taking the averages of the current classes (clusters)
4. Do 2-3 until centroid are stabilized.

Caveats for K-means:

* Although it can be proved that the procedure will always terminate, the k-means algorithm does not necessarily find the most optimal configuration, corresponding to the global objective function minimum.
* The algorithm is also significantly sensitive to the initial randomly selected cluster centres. The k-means algorithm can be run multiple times to reduce this effect.

Here is the basic animation to show the intuition of K-means.

![](http://shabal.in/visuals/kmeans/left.gif)

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Process of defining a machine learning solution (ML#2)](http://www.erogol.com/process-of-defining-a-machine-learning-solution-ml2/ "Process of defining a machine learning solution (ML#2)")
2. [Anomaly detection and a simple algorithm with probabilistic approach.](http://www.erogol.com/anomaly-detection-and-a-simple-algorithm-with-probabilistic-approach/ "Anomaly detection and a simple algorithm with probabilistic approach.")
3. [Hinton's NN journey](http://www.erogol.com/hintons-nn-journey/ "Hinton's NN journey")
4. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")