---
layout: post
title: "ML Work-Flow (Part 3) - Feature Extraction"
description: "In this post, I'll talk about the details of Feature Extraction (aka Feature Construction, Feature A"
tags: classification data mining deep learning feature extraction feature learning
minute: 9
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

In this post, I'll talk about the details of **Feature Extraction** (aka Feature Construction, Feature Aggregation …) in the path of successful ML. Finding good feature representations is a domain related process and it has an important influence on your final results. Even if you keep all the settings same, with different Feature Extraction methods you would observe drastically different results at the end. Therefore, choosing the correct Feature Extraction methodology requires painstaking work.

**Feature Extraction** is a process of conveying the given raw data into set of instance points embedded in a **standardized**, **distinctive** and **machine understandable** space. Standardized means comparable representations with same length; so you can compute similarities or differences of the instances that have initially very versatile structural differences (like different length documents). Distinctive means having different feature values for different class instances so that we can observe clusters of different classes in the new data space. Machine understandable representation is mostly the numerical representation of the given instances. You can understand any document by reading it but machines only understand semantics implied by the numbers.

Initially, we can divide Feature Extraction into two main sub-headings;

### **Hands o****n** **approach**

This is what the expert data analysts generally do; the process of discovering features with the mixture of expert knowledge, data analysis and inference of the analytic observations. This is an important ability especially for the industry where automatized solutions are not very helpful in relation to certain disciplines such as NLP and Computer Vision. We know what is ultimately important and what kind of regularities are implicitly involved in these domains. However, given a raw database of customers with transactional histories, what are the important features that direct particular set of customers to certain behaviors is a complete mystery. This requires human intuition, expert knowledge or at least a human hand to infer the numbers in a semantic manner.

Hands of approach is the iteration of analysis, inference, hypothesis, testing and recursion. You analyze the data, claim some commonality between instances, test your hypothesis with statistical tools, if true add it to your approved feature set otherwise re-define your claim or ignore it. (I guess this requires a separate blog post for more discussion.)

In **[SENTIO](http://sentiosports.com/)** **SPORTS**, I am also trying to do the same for soccer teams and players. For instance, we try to predict outcomes of any future match based on computed features of the teams and players. However, these features are not automated and I try to analyze historical data to come up with touchstone values, correlation and causation effects between different values. This is like seeking for a needle in a haystack, especially if you are not very competent in Soccer (like me 🙂 but I believe in numbers!)

### **Algorithmic Methods**

#### **D******eterministic Algorithms****:

If we are working on a certain discipline that we know the important aspects that separates one instance from the other, than we can find a set of deterministic rules to discover these. In many fields like Computer Vision, this is the case and people continuously come up with new algorithms. These methods usually rely on some researches which subjected human perception and cognition. As soon as researchers provide some clues of human visual perception based on edges and corners on the images, Computer Vision community devices algorithms discovering same structures on the given images and converting these structures into numerical forms. For example, they count number of edges in certain orientations and create histograms by these numbers. These are deterministic methods in that aspect, if you provide the same data you get the same feature values.

I am mostly experienced in Feature Learning and Computer Vision but I will try to summarize some known Feature Extraction algorithms for particular fields;

* **NLP**
  + **Bag of Words:** This is very simple yet very powerful Feature Extraction method. It is simply finding important set of words in a given corpus (called vocabulary), then counting these in each document and creating a histogram of word frequencies for each document. Many sentiment analysis, document classification application still use BoW as a feature extraction method.
  + **N-Grams:** Instead of taking each word as a single unit, include some level of combinational information and consider word groups. ([Wikipedia](http://en.wikipedia.org/wiki/N-gram))
  + **Feature Hashing:** Even though it is more common as a post-processing following Feature Extraction to increase efficiency, it can also be defined as a Feature Extraction method by itself. The idea is to apply basic hashing tricks to given data to extract features. Thus, we believe that any similar set of items will have similar hash values.
* ****Computer Vision****
  + ****SIFT******:** Scale Invariant Feature Transform is maybe the most common Feature Extraction algorithm, especially in the industrial applications. It is basically the combination of Image Scaling + Edge Detection at different scales + Finding Region of Interests + Histogram of Different orientation ROIs. ([best tutorial for SIFT](http://www.aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/))

    ![](http://www.codeproject.com/KB/recipes/619039/SIFT.JPG)

    SIFT - Scale Invariant Feature Transform
  + ****HOG:**** Histogram of Oriented Gradients is the state of art for pedestrian detection. Very similar to SIFT with little technical differences like contrast normalization.

    ![](http://www.mathworks.com/help/vision/ug/traincascadeobjectdetector_bicyclehog.png)

    HOG - Histogram of Oriented Gradients
  + ****LBP:**** Local Binary Patterns is the easiest and the fastest way of getting textural definition of an image and it has very successful face recognition applications.

    ![](http://www.mathworks.com/matlabcentral/fileexchange/screenshots/7405/original.jpg)

    LBP - Local Binary Patterns
  + ****Bag of**** ****Words:**** This is BoW application to image domain with defined visual words by SIFT, HOG or LPB of the given image patches. ([tutorial](http://www.robots.ox.ac.uk/~az/icvss08_az_bow.pdf))

    ![](http://gilscvblog.files.wordpress.com/2013/08/figure31.jpg)

    Bag of Words for images
* ****Sound Recognition**** (I am not able to dig more but I know [that toolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) with great feature versatility)
  + ****MFCC****
  + ****LPC****
  + ****PLP****

#### **Feature** **(Representation)** **Learning Algorithms:**

Ohhh Man! This is where the things are changing right now. Feature Learning is to learn transformations of raw instances to representative and discriminative representations that are useful for any further supervised or unsupervised purpose. As the name suggests, in these set of methods we learn the the representations as well as the final prediction models. If we ask why this is useful, here are some bullet-points;

* **Agnostic** **Application****:** Same learning algorithm can learn features for different domains like images or texts with little or no changes.
* **Domain Adaptation :**You can train one model and use it for many different datasets from different resources. It also gauges the domain-shift problem by better generalization performance. (Domain-shift is explained as the statistical differences of two datasets from different data resources, hence any model learned from one dataset can give poor results for the other). There are many practical achievements of this approach. People in Computer Vision community uses pre-trained neural networks at Image-Net for any other classification tasks even the target concept are not involved in Image-Net. I believe this is a very solid step forward to true AI.
* **Multi-Task** **learning** : Learned features can be used for any objective such as retrieval and classification.
* **Multi-Sensory input :** With little preprocessing, you can provide different sensory data to the single model at the same time and learn cooperated representations at the end. For instance, Sound and Text data can be provided at the same time to learn one single representation for each instance.

There is a high research activity going on for Feature Learning at the present era of science, especially with the attention of huge tech companies like Microsoft, Facebook, Google. It also accounts for a sub-topic for Deep Learning community.

Although there are vast amount of different **feature learning algorithms,** I list some of them here:

* **Deep Learning**
  + Auto-Encoder
  + Restrictive Boltzman Machines
  + Convolutional Neural Networks
  + De-convolutional Neural Networks

    ![](http://ufldl.stanford.edu/wiki/images/thumb/5/5c/Stacked_Combined.png/500px-Stacked_Combined.png)

    Neural Network - Each layer includes different level of feature values.
* **Sparse Coding** ([LeCun's works](http://cs.nyu.edu/~yann/research/sparse/))
  + Compressed Sensing (precursor of Auto-Encoders)
  + Structural Sparsity Algorithms
  + Recursive Sparse Coding (theoretically equivalent to Neural Networks)

    ![](http://www.ifp.illinois.edu/~yuhuang/SparsityCoding/sparsecoding.jpg)

    Sparse coding
* **Matrix Factorization** ([a basic tutorial](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/))
  + PCA
  + SVD

### **Practical Concerns**

Okay, we talked much about Feature Extraction in many different perspectives like algorithms to general methodologies. Now, I'll talk a bit about the practical concerns of Feature Extraction step.

You are given a “crowd” of data and expecting to extract the “best” features “possible” for your ultimate “purpose”. I quoted important aspects that shape your choice.

* **Crowd**: How large is your data. You might know the best algorithm but if it is not scalable with your data, it is meaningless and even time-consuming. You should consider computational issues in the prospected system.
* **Best**: Probe for the state of art and do some research about the target problem. Without any research, just by hearsay arguments, results are likely to be disappointing.
* **Possible:** You find the state of art Feature Extraction method but you are not able to understand it. Choose the another one that you can understand all the bits and pieces or work hard to lean this. Many people tend to use algorithms just by calling the related function without any intellectual understanding. I believe this is a very serious mistake. Because you are getting into a dark road that you cannot explain in the further steps. You should know all the details, especially desired input and expected outputs of the Feature Extraction algorithm so that you can check the correctness and the understand the meaning of each feature value.
* **Purpose:** Why you need these features. That is the other fundamental question. Some set of features might work well for classification, yet the another for retrieval.

As a very stupid but important side note, “**CHECK FOR NULL!!!**”. After each Feature Extraction process, do not forget to check all your data for NULL values. Otherwise, you find yourself in very hazy situations that take your days with a single NULL value .

This is all for now. During the week, if I find any further comment or important stuff about Feature Extraction, I'll also include them here. Thanks for watching 🙂

[Share](https://www.addtoany.com/share)

### Related posts:

1. [A Large set of Machine Learning Resources for Beginners to Mavens](http://www.erogol.com/large-set-machine-learning-resources-beginners-mavens/ "A Large set of Machine Learning Resources for Beginners to Mavens")
2. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
3. [ML Work-Flow (Part 4) – Sanity Checks and Data Splitting](http://www.erogol.com/ml-work-flow-part-4-sanity-checks-data-spliting/ "ML Work-Flow (Part 4) – Sanity Checks and Data Splitting")
4. [ML Work-Flow (Part 5) – Feature Preprocessing](http://www.erogol.com/ml-work-flow-part-5-feature-processing/ "ML Work-Flow (Part 5) – Feature Preprocessing")