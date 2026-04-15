---
layout: post
title: "ML WORK-FLOW (Part2) - Data Preprocessing"
description: "I try to keep my promised schedule on as much as possible"
tags: data discretization data science data_cleaning data_integration data_mining
minute: 5
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I try to keep my promised schedule on as much as possible. Here is the detailed the first step discussion of my proposed [Machine Learning Work-Flow](http://www.erogol.com/machine-learning-work-flow-part-1/), that is Data Preprocessing.

Data Preprocessing is an important step in which mostly aims to improve raw data quality before you dwell into the technical concerns. Even-though this step involves very easy tasks to do, without this, you might observe very false or even freaking results at the end.

I also stated at the work-flow that, Data Preprocessing is statistical job other than ML. By saying this, Data Preprocessing demands good data inference and analysis just before any possible decision you made. These components are not the subjects of a ML course but are for a Statistics. Hence, if you aim to be cannier at ML as a whole, do not ignore statistics.

We can divide Data Preprocessing into 5 different headings;

1. Data Integration
2. Data Cleaning
3. Data Transformation
4. Data Discretization
5. Data Reduction

### **Data Integration**

**Put different format data from various sources into** a uniform shape suitable for the upcoming processes. These different sources might be called different databases, streams even excel tables. Albeit the simplicity of the idea, this emerges a different set of commercial softwares, namely ETL (Extract - Transform - Load ) tools. These tools make you able to reach different sources from a single point of view and merge data with defined homogenize data-flow. Incidently, data integration includes the other headings in itself recursively. More explicitly, any sub-component of your integration flow is able to include one of more Data Preprocessing process that we explain below.

It is important to **define data format** without any hesitation in advance, according to your problem. If you are not very sure about the convenient format, investigate it. Otherwise, integration might be too obscure . It is very time consuming, especially for big data, and motivation breaking for the next steps.

### **Data Cleaning**

**Fill the missing values** in the data, attributes or class labels.  Most simple approach is to use mean or median value of the other rows or mean or median of same class instances. (Median is robust to outlier values in general) . Maybe the other approach is to train a model for the prediction of the missing values just like the class labels.

**Identify outliers and smooth out noisy instances.** Outliers and noisy instances are deceiving for many ML algorithms like AdaBoost. Therefore, you need to rectify the data before any further proceeding. Even, you need to repeat all Preprocessing again after you remove outliers since , for instance, if you fill the missing values by including outliers, these are also wrong and need to be re-defined.

For outlier removal, one common way is to Cluster the data and remove the poor clusters. Moreover, you can use particular outlier detection algorithm (such as my baby [RSOM](http://www.erogol.com/eccv2014-work-conceptmap-mining-noisy-web-data-concept-learning/) or LOF). Another option is to fit a Regression model and align the data into this to remove the outlier effect.

**Correct inconsistency in the data**. This requires expert knowledge in general. You should consult to you business partner or the customer.

### **Data Transformation**

**Normalization -Scaling - Standardization**. Depending on your further steps like feature extraction, you may need to transform data into different scales or domains.  This is very brutally important to get high quality, discriminative features at the end. Especially, if you are using automatized feature extraction algorithms, in general, they expects certain data formats and they are very fragile about it.

**Construct new attributes**. For instance, if you have weight and height values of the customers, adding  BMI as a new attribute is very reasonable. Such attribute constructions need some level of experience and statistical knowledge in the domain but creates very big performance improvements.

### **Data Discretization**

Continuous values are problematic in some cases and for particular ML algorithms. Even I try to avoid discretizing data with reasonable algorithms, specially for inference purposes discretization is very essential.

**Use unsupervised equal-binning**. Divide the numeric data into equal size or range bins without any detailed considerations.

**Supervised discretization**. Use class boundaries by sorting values and placing hinges between values by observing class distributions on the values.  You can also use entropy measure to define the partition. Now, you defines some candidate set of value partitions but you decide the best one with the best entropy based,information gain value.  My choice is to use continuous value capable Decision Tree to define value partitions from the nodes of the constructed tree.

### **Data Reduction**

**Reduce number of instances**. Sometimes, you prefer to use subset of the data instead of the whole junk. In that case, sampling schema works for you. Even though, there are many different sampling methods, I prefer the most naive one, Random Sampling. If I need more robust results with multiple subset, I prefer to use bootstrapping with replacement.

**Reduce number of attributes.**Please do not try to predict number of Nobel Prizes of a country by the chocolate consumption (This is real story).

![](http://blogs.scientificamerican.com/the-curious-wavefunction/files/2012/11/Screen-Shot-2012-11-20-at-4.46.58-PM1.png)

Nobel Prize vs Chocolate Consumption

Although this needs some level of expertise, if you are sure about any irrelevant attribute remove it from your attribute lists. However, if you are hesitated then wait for the Feature Selection step for its magic.

As a side-note, there is also a sub-topic in ML that applies reduction paradigm to complex problems so that it can solve the whole problem by coming through the simple sub-problems. [MORE...](http://blogs.scientificamerican.com/the-curious-wavefunction/files/2012/11/Screen-Shot-2012-11-20-at-4.46.58-PM1.png)

[Share](https://www.addtoany.com/share)

### Related posts:

1. [ML Work-Flow (Part 5) – Feature Preprocessing](http://www.erogol.com/ml-work-flow-part-5-feature-processing/ "ML Work-Flow (Part 5) – Feature Preprocessing")
2. [What is special about rectifier neural units used in NN learning?](http://www.erogol.com/what-is-special-about-rectifier-neural-units-used-in-nn-learning/ "What is special about rectifier neural units used in NN learning?")
3. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
4. [Machine Learning Work-Flow (Part 1)](http://www.erogol.com/machine-learning-work-flow-part-1/ "Machine Learning Work-Flow (Part 1)")