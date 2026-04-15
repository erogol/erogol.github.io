---
layout: post
title: "Ensembling Against Adversarial Instances"
description: " What is Adversarial?

Machine learning is everywhere and we are amazed with capabilities of these a"
tags: adversarial convnet deep learning ensemble image recognition
minute: 9
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

### What is Adversarial?

Machine learning is everywhere and we are amazed with capabilities of these algorithms. However, they are not great and sometimes they behave so dumb.  For instance, let's consider an image recognition model. This model  induces really high empirical performance and it works great for normal images. Nevertheless, it might fail when you change some of the pixels of an image even so this little perturbation might be indifferent to human eye. There we call this image an adversarial instance.

There are various methods to generate adversarial instances [1][2][3][4]. One method is to take derivative of the model outputs wrt the input values so that we can change instance values to manipulate the model decision. Another approach exploits genetic algorithms to generate manipulative instances which are confidently classified as a known concept (say 'dog') but they are nothing to human eyes.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/adver_1-1.png)

Generating adversaries by genetic algorithm [1]

### 

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/adver_2.png)

Generating adversaries by input gradient [2].

So why these models are that weak against adversarial instances. One reliable idea states that because adversarial instances lie on the low probability regions of the instance space. Therefore, they are so weird to the network which is trained with a limited number of instances from higher probability regions.

That being said, maybe there is no way to escape from the fretting adversarial instances, especially when they are produced by exploiting weaknesses of a target model with a gradient guided probing. This is a analytic way of searching for a misleading input for that model with an (almost) guaranteed certainty. Therefore in one way or another, we find an perturbed input deceiving any model.

Due to that observation, I believe that adversarial instances can be resolved by multiple models backing each other. In essence, this is the motivation of this work.

### Proposed Work

In this work, I like to share my observations focusing on strength of the ensembles against adversarial instances. This is just a toy example with so much short-comings but I hope it'll give the idea with some emiprical evidences.

As a summary, this is what we do here;

* Train a baseline MNIST ConvNet.
* Create adversarial instances on this model by using [cleverhans](https://github.com/openai/cleverhans) and save.
* Measure the baseline model performance on adversarial.
* Train the same ConvNet architecture including adversarial instances and measure its performance.
* Train an ensemble of 10 models of the same ConvNet architecture and measure ensemble performance and support the backing argument stated above.

My code full code can be seen on [github](https://github.com/erogol/StudyAdversarials) and I here only share the results and observations. You need [cleverhans](https://github.com/openai/cleverhans), Tensorflow and Keras for adversarial generation and you need PyTorch for ensemble training. (Sorry for verbosity of libraries but I like to try PyTorch as well after yeras of tears with Lua).

One problem of the proposed experiment is that we do not recreate adversarial instances for each model and we use a previously created one. Anyways, I believe the empirical values verifies my assumption even in this setting.  In addition,  I plan to do more extensive study as a future work.

#### Create adversarial instances.

I start by training a simple ConvNet architecture on MNIST dataset by using legitimate train and test set splits. This network gives 0.98 test set accuracy after 5 epochs.

For creating adversarial instances, I use fast gradient sign method which perturbs images using the derivative of the model outputs wrt the input values.  You can see a bunch of adversarial samples below.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/sample_adversarials.png)

The same network suffers on adversarial instances (as above) created on the legitimate test set. It gives 0.09 accuracy which is worse then random guess.

#### Plot adversarial instances.

Then I like to see the representational power of the trained model on both the normal and the adversarial instances. I do this by using well-known dimension reduction technique T-SNE. I first compute the last hidden layer representation of the network per instance and use these values as an input to T-SNE which aims to project data onto 2-D space. Here is the final projection for the both types of data.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/nn_mnist_normal_instances_tsne.png)

Projection of normal test set.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/nn_mnist_adversarial_instances_tsne.png)

Projection of adversarial instances.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/02/nn_mnist_all_instances_tsne.png)

Projection of both adversarial and normal test instances.

These projections clearly show that adversarial instances are just a random data points to the trained model and they are receding from the real data points creating what we call low probability regions for the trained model. I also trained the same model architecture by dynamically creating adversarial instances in train time then test its value on the adversarials created previously. This new model yields 0.98 on normal test set, 0.91 on previously created adversarial test set and 0.71 on its own dynamically created adversarial.

Above results show that including adversarial instances strengthen the model. However,  this is conforming to the low probability region argument. By providing adversarial, we let the model to discover low probability regions of adversarial instances. Beside, this is not applicable to large scale problems like ImageNet since you cannot afford to augment your millions of images per iteration. Therefore,  by assuming it works, ensembling is more viable alternative as already a common method to increase overall prediction performance.

#### Ensemble Training

In this part, I train multiple models in different ensemble settings. First, I train N different models with the same whole train data. Then, I bootstrap as I train N different models by randomly sampling data from the normal train set. I also observe the affect of N.

The best single model obtains 0.98 accuracy on the legitimate test set. However, the best single model only obtains 0.22 accuracy on the adversarial instances created in previous part.

When we ensemble models by averaging scores, we do not see any gain and we stuck on 0.24 accuracy for the both training settings. However, surprisingly when we perform max ensemble (only count on the most confident model for each instance), we observe 0.35 for uniformly trained ensemble and 0.57 for the bootstrapped ensemble with N equals to 50.

Increasing N raises the adversarial performance. It is much more effective on bootstrapped ensemble. With N=5 we obtain 0.27 for uniform ensemble and 0.32 for bootstrapped ensemble. With N=25 we obtain 0.30 and 0.45 respectively.

These values are interesting especially for the difference of mean and max ensemble. My intuition behind the superiority of maxing is maxing out predictions is able to cover up weaknesses of models by the most confident one, as I suggested in the first place. In that vein, one following observation is that adversarial performance increases as we use smaller random chunks for each model up to a certain threshold with increasing N (number of models in ensemble). It shows us that bootstrapping enables models to learn some of the local regions better and some worse but the worse sides are covered by the more confident model in the ensemble.

As I said before, it is not convenient to use previously created adversarials created by the baseline model in the first part. However, I believe my claim still holds. Assume that we include the baseline model in our best max ensemble above. Still its mistakes would be corrected by the other models. I also tried this (after the comments below) and include the baseline model in our ensemble. 0.57 accuracy only reduces to 0.55. It is still pretty high compared to any other method not seeing adversarial in the training phase.

#### Conclusion

1. It is much more harder to create adversarials for ensemble of models with gradient methods. However, genetic algorithms are applicable.
2. Blind stops of individual models are covered by the peers in the ensemble when we rely on the most confident one.
3. We observe that as we train a model with dynamically created adversarial instances per iteration, it resolves the adversarials created by the test set. That is, since as the model sees examples from these regions it becomes immune to adversarials. It supports the argument stating low probability regions carry adversarial instances.

### (Before finish) This is Serious!

Before I finish, I like to widen the meaning of this post's heading. Ensemble against adversarial!!

"Adversarial instances" is peculiar AI topic. It attracted so much interest first but now it seems forgotten beside research targeting GANs since it does not yield direct profit, compared to having better accuracy.

Even though this is the case hitherto, we need consider this topic more painstakingly from now on. As we witness more extensive and greater AI in many different domains (such as health, law, governace), adversarial instances akin to cause greater problems intentionally or by pure randomness. This is not a sci-fi scenario I'm drawing here. It is a reality as it is prototyped in [3]. Just switch a simple recognition model in [3]  with a [AI ruling court for justice](http://www.telegraph.co.uk/science/2016/10/23/artifically-intelligent-judge-developed-which-can-predict-court/).

Therefore, if we believe in a future embracing AI as a great tool to "make the world better place!", we need to study this subject extensively before passing a certain AI threshold.

#### Last Words

This work overlooks many important aspects but after all it only aims to share some of my findings in a spare time research.  For a next post, I like study unsupervised models like Variational Encoders and Denoising Autoencoders by applying these on adversarial instances (I already started!). In addition, I plan to work on other methods for creating different types of adversarials.

From this post you should take;

* References to adversarial instances
* Good example codes waiting you on [github](https://github.com/erogol/StudyAdversarials) that can be used many different projects.
* Power of ensemble.
* Some of non-proven claims and opinions on the topic.

IN ANY WAY HOPE YOU LIKE IT ! 🙂

**References**

[1] Nguyen, A., Yosinski, J., & Clune, J. (2015). Deep Neural Networks are Easily Fooled. Computer Vision and Pattern Recognition, 2015 IEEE Conference on, 427–436.

[2] Szegedy, C., Zaremba, W., & Sutskever, I. (2013). Intriguing properties of neural networks. *arXiv Preprint arXiv: …*, 1–10. Retrieved from http://arxiv.org/abs/1312.6199

[3] Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2016). Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples. *arXiv*. Retrieved from http://arxiv.org/abs/1602.02697

[4] Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. *Iclr 2015*, 1–11. Retrieved from http://arxiv.org/abs/1412.6572

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Selfai: A Method for Understanding Beauty in Selfies](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/ "Selfai: A Method for Understanding Beauty in Selfies")
2. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
3. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
4. [Why do we need better word representations ?](http://www.erogol.com/need-better-word-representations/ "Why do we need better word representations ?")