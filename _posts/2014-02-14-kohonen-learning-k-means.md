---
layout: post
title: "Kohonen Learning Procedure K-Means vs Lloyd's K-means"
description: "K-means maybe the most common data quantization method, used widely for many different domain of pro"
tags: clustering code github kmeans kohonen
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

K-means maybe the most common data quantization method, used widely for many different domain of problems. Even it relies on very simple idea, it proposes satisfying results in a computationally efficient environment.

Underneath of the formula of K-means optimization, the objective is to minimize the distance between data points to its closest centroid (cluster center). Here we can write the objective as;

![argmin sum_{i=1}^{k}sum_{x_j in S_i} ||x_j - mu_i||^2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_a00518086023705ec8c53b62214e3260.gif)argmin sum\_{i=1}^{k}sum\_{x\_j in S\_i} ||x\_j - mu\_i||^2

![mu_i](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_ea70175a925b13009ea2178ac484d724.gif)mu\_i is the closest centroid to instance ![x_j](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_1f89889020cdc84d9e1c35237cb62f65.gif)x\_j.

One of the footnote for Lloyd's above K-means formula, it implicitly enforces similar size clusters, although it is ill-suited assumption for many of the data-sets. In addition, since we compute new centroids over the whole data for each iteration, there is almost nothing randomized so objective is intended to stay onto same optimal point of the objective with multiple successive runs..

In order to deface the levelled problems of K-means, one alternative approach is to use Kohonen's Learning Procedure (KLP) instead of mean quantization. It brings more direct optimization steps (delta rule), controlled by a learning rate alpha. It also conforms to randomized batch or a online (stochastic) learning better in relation to original K-means, proposing possible  improvements to same optimal point and same size cluster problems.  In that manner, It is prone to better mapping of the data-set over the clusters. In addition, from my observations, it is faster to reach a optimal point. It is very essential feature, especially for large scale problems.

KLP stochastic learning optimization is formulated as ;

![w_j^{t+1} = w_j^t + alpha (x_i - w_j^t)](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_c55c53c204a29e3ded43d42c6a5a23e6.gif)w\_j^{t+1} = w\_j^t + alpha (x\_i - w\_j^t)

superscript ![t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_e358efa489f58062f10dd7316b65649e.gif) is the iteration count, ![w_j](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_e8100be07fa5419af6c6738b934dfca0.gif)w\_j is the closest cluster centroid to given instance vector ![x_i](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_1ba8aaab47179b3d3e24b0ccea9f4e30.gif)x\_i.  The idea is very similar to Lloyd's, in essence. For a given instance, its closest centroid is found and weight vector of that centroid is updated toward the instance vector. For batch version, the formula is same but the update step is defined by the mean of the distances between the centroid and its matching instances. Basically, the difference here is to use the distance between instances and the present centroids, contrary to mean heuristic of Llyod.

[KLP-Kmeans](https://github.com/erogol/KLP_KMEANS/blob/master/klp_kmeans.py "KLP-Kmeans github") including three alternative implementations as Scipy, Theano, Theano-GPU, is at my [Github Repository](https://github.com/erogol/KLP_KMEANS/blob/master/klp_kmeans.py) with simple demonstrations. Even Scipy version is faster for small scale problems as the scale of matrix operations is getting larger Theno contrives callous improvements. Furthermore, with increasing batch size GPU based Theano implementation should be the choice.



### Related posts:

2. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
3. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")