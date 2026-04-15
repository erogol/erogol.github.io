---
layout: post
title: "Anomaly detection and a simple algorithm with probabilistic approach."
description: "What is anomaly detection?It is the way of detecting a outlier data point among the other points tha"
tags: algorithm anomaly detection machine learning tutorial
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**What is anomaly detection?**It is the way of detecting a **outlier data** point among the other points that have a some kind of logical distribution.  **Outlier** one is also **anomalous** point (**Figure 1**)

[![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/10/anomaly_fig1.jpg "anomaly_fig1")](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/10/anomaly_fig1.jpg)

Figure 1

**What are the applications?**

* ***Fraud user activity detection*** - it is a way of detecting hacker activities on web applications or network connections by considering varying attributes of the present status. For example , an application can keep track of the user's inputs to website and the work load that he proposes to system. Considering these current attribute values detection system decide a particular fraud action and kick out the user if there is.
* ***Data center monitoring**-* You might be governing a data center with vast amount of computers so it is really hard to check each computer regularly for any flaw. A anomaly detection system might be working by considering network connection parameters of the computers, CPU and Memory Loads, it detect any problem on computer.

# **General procedure for anomaly detection**

We have a dataset that shows the data instances with some corresponding attributes without any anomalies. With that data set it is possible to **create a model** to represent these **regular instances**. Then, any **given instance** can be **compared** with that model and if it is not fitting with the model in some degree, flag that instance as anomalous instance.

### **Probabilistic approach to method**

First of all we need to know **Gaussian** (Normal) **Distribution** (is a preliminary subject for statistics and probabilistic machine learning). Gaussian points distribution symmetric around the mean value and spread with respect to the variance so it has two parameters as **mean** *μ*and **variance ![\sigma^2\,](http://upload.wikimedia.org/math/2/4/d/24dd4eca5f79ddd3740ac274afded971.png)**. These two parameters are enough to define a gaussian. (Figure 2)

![](http://www.jlplanner.com/html/stddev.gif)

Lets get into the algorithm. General sense of the algorithm is to **find a Gaussian** Distribution **over each attribute** of the data and **look the standing of new data** on these distributions. If it is standing awkwardly in overall, flag it as anomalous instance.

As we talked we need to have variance and mean to define a distribution over attributes. For each attribute on dataset find these.

mean of attribute i  = (1/n)\*sum all attr Xi

variance of attribute i = (1/n)\*(mean of attribute i - Xi)^2

n = total number of rows.

After we find mean and variance or each attribute on dataset, assume you have new instance Xm with attributes {x1,x2,x3,...,xk} look for                                                     **P(Xm) = product of all p(xi;mean i, variance i)**.

If **P(Xm) is smaller than a given threshold ε** flag Xm as anomalous.

Intuitional explanation,  we computed the probability of Xm being a member of our  seen data set.

**Caveats for implementation**

**Caveat 1**

It is possible to have attributes not in Gaussian Dist. . There are couple of ways to converge them to Gaussian.

* Take **log()** of all the values of attribute ---- X ---> log(X)
* Take **root** of the all attribute values ---- X ---> X^(-1/2)

[![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/10/log.png "log")](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/10/log.png)

Figure 3

**Caveat 2**

Do we need to use all attributes comes with data set? We need to be selective on attribute selection. Selected ones need to be good selectors for anomalous instances. One way to see this is to draw a graph that shows the **standing of instances** on single attribute and sign the anomalous ones on the graph. Anomalous instances need to be **away** from the mean of the distribution that is attribute is a good selector for anomalies.

**What is different, Supervised Learning vs Anomaly Detection**

Anomaly detection is used for

* If you have a data set , **rich** for regular instances and **poor** for anomalous instances
* You are learning the model generalize for regular instances by evaluating lots of regular instances and check the new instance whether it is one of the regulars or not.

Supervised Learning is used in case

* You have **balanced** number of regular and anomalous instances in training set.
* You can generalize both for regular and anomalous instances.



---

**Related posts:**

1. [Process of defining a machine learning solution (ML#2)](http://www.erogol.com/process-of-defining-a-machine-learning-solution-ml2/ "Process of defining a machine learning solution (ML#2)")
2. [a 30 min good presentation for Machine Learning](http://www.erogol.com/a-30-min-good-presentation-for-machine-learning/ "a 30 min good presentation for Machine Learning")
3. [Some Basic Machine Learning Terms #1](http://www.erogol.com/some-basic-machine-learning-terms-1/ "Some Basic Machine Learning Terms #1")
4. [How K-means clustering works](http://www.erogol.com/how-k-means-clustering-works/ "How K-means clustering works")