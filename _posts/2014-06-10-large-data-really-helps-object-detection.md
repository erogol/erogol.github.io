---
layout: post
title: "Large data really helps for Object Detection ?"
description: "I stumbled upon a interesting BMVC 2012 paper (Do We Need More Training Data or Better Models for Ob"
tags: big data computer vision machine learning paper review
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I stumbled upon a interesting BMVC 2012 paper (***Do We Need More Training Data or Better Models for Object Detection?*** -- Zhu, Xiangxin, Vondrick, Carl, Ramanan, Deva, Fowlkes, Charless). It is claming something contrary to current notion of big data theory that advocates benefit of large data-sets so as to learn better models with increasing training data size. Nevertheless, the paper states that large training data is not that much helpful for learning better models, indeed more data is maleficent without careful tuning of your system !!

The paper highlights important problems of standard object recognition (HOG + Linear SVM) pipeline with large training data;

* Aiming direct solid performance from models learned from large training data is not feasible. The real gain is hidden behind the noise-proof and structured train data.
* SVM is sensitive to noise with its unbounded hinge penalty. Outlier instances are able to budge the decision boundary wildly.
* Even if you set your pipeline with correct set of steps, the performance saturates after some level of training data.

[![fig1](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/06/fig1.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/06/fig1.png)

This figure shows the effect of clean data (middle ) vs large but noisy data model (right). Right shows the SVM model learned from all aspects of the face images whereas Middle shows the model of only frontal face images with better performance on the test set.

[![PASCAL performance vs number of training instances](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/06/fig2.png)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2014/06/fig2.png)

PASCAL performance vs number of training instances

They propose some ideas to evade the pointed problems and utilize large training data in a correct manner. The first thing is to tune your regularizer  C by  cross-validation on different training data sizes. They state that C values tuned in that way give better performance with larger training data-set.

Another fact is to capture sub-modularity of the given data. For this aim, they cluster the training images and learn separate models from each cluster. However, there are some constraints of this approach. First, you need to balance the number of clusters you demand and the number of training instances. Dividing insufficient data aggressively  or  plenty amount of data parsimoniously, both degrades the performance since in the first case your models are too general and in the second case models are over-fitted (SVM is very likely to over-fit data). Despite of these caveats, capturing sub-modularity is able to enhance the performance with moderate balance of clusters and the data size.

They also highlight part based models as implicit mixture models. Each different combination of parts corresponds to different sub-modularity of the target category. Therefore these model are supposedly more expressive with larger data-sets.

Their arguments might be enlisted as below;

* Accord your regularization parameter correctly relative to different number of training instances.
* If data-set is large enough to have clean set of clusters, delicately use sub-modularity of data.
* Part based models are expressive by implicit sub-modularity.
* Experiments shows that, even with correct settings, performance saturates after some number of training instances. Therefore, instead of learning more advance models, future works should focus on more expressive feature descriptors (they believe that HOG has clear limitations) and compositional approaches that uses finer sub-modular information lying behind the data-set.

I also need to criticize this work by being too limited to HOG+Linear SVM. They only infer a single method to conclude big data is more than enough in the most cases. However, in the recent literature of Object Recognition and Detection researchers are able to show better and better results approaching human performance by the goods of big data. Especially Deep Learning architectures or Semi-Supervised algorithms are very promising in that direction. Therefore, in that limited score of the paper, proposed arguments seem arguable. However, I personally like to see such propositional works that ask interesting questions.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [How many training samples we observe over life time ?](http://www.erogol.com/1301-2/ "How many training samples we observe over life time ?")
2. [What SVM does with a video](http://www.erogol.com/what-svm-does-with-a-video/ "What SVM does with a video")
3. [Our ECCV2014 work "ConceptMap: Mining noisy web data for concept learning"](http://www.erogol.com/eccv2014-work-conceptmap-mining-noisy-web-data-concept-learning/ "Our ECCV2014 work \"ConceptMap: Mining noisy web data for concept learning\"")
4. [Recent Advances in Deep Learning](http://www.erogol.com/recent-advances-in-deep-learning/ "Recent Advances in Deep Learning")