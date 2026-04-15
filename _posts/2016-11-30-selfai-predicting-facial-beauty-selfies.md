---
layout: post
title: "Selfai: A Method for Understanding Beauty in Selfies"
description: " 

Selfies are everywhere"
tags: cnn deep learning image recognition research notes selfai
minute: 13
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

### 

Selfies are everywhere. With different fun masks, poses and filters,  it goes crazy.  When we coincide with any of these selfies, we automatically give an intuitive score regarding the quality and beauty of the selfie. However, it is not really possible to describe what makes a beautiful selfie. There are some obvious attributes but they are not fully prescribed.

With the folks at 8bit.ai, we decided to develop a system which analyzes selfie images and scores them in accordance to its quality and beauty.  The idea was to see whether it is possible to mimic that bizarre perceptual understanding of human with the recent advancements of AI. And if it is, then let's make a mobile application and let people use it for whatever purpose. Spoiler alert! We already developed [Selfai app](http://selfai-app.8bit.ai) available on iOS and Android and we have one instagram bot [@selfai\_robot](https://www.instagram.com/selfai_robot/). You can check before reading.

![Selfai - available on iOS and Android](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/selfai.jpg)

Selfai - available on iOS and Android

After a kind of self-promotional entry, let's come to the essence. In this post, I like to talk about what I've done in this fun project from research point. It entails to a novel method which is also applicable to similar fine-grain image recognition problems beyond this particular one.

I call the problem fine-grain since what differentiates the score of a selfie relies on the very details. It is hard to capture compared to the traditional object categorization problems, even with simple deep learning models.

We like to model 'human eye evaluation of a selfie image' by a computer. Here; we do not define what the beauty is, which is a very vague term by itself, but let the model internalize the notion from the data. The data is labeled by human annotators on an internally developed crowd-sourced website.

In terms of research, this is a peculiar problem where traditional CNN approaches fail due to following reasons:

* Fine-grain attributes are the factors defining one image better or  worse  than another.
* Selfie images induce vast amount of variations with different applied filters, editions, pose and lighting.
* Scoring is a different practice than categorization and it is not a well-studied problem compared to categorization.
* Scarcity of annotated data yields learning in a small-data regime.

### Previous Works

This is a problem already targeted by different works. [HowHot.io](http://www.howhot.io/) is one of the well-known example of such, using deep learning back-end empowered with a large amount of data from a dating application. They use the application statistics as the annotation. Our solution differs strongly since we only use in-house data which is very small compared to what they have. Thus feeding data into a well-known CNN architecture simply does not work in our setting.

There is also a relevant [blog post](http://karpathy.github.io/2015/10/25/selfie/) by A. Karpathy where he crawled Instagram for millions of images and use "likes" as annotation. He uses a simple CNN. He states that the model is not that good but still it gives a intuition about what is a good selfie. Again, we count on A. Karpathy that ad-hoc CNN solutions are not enough for decent results.

There are other research efforts suggesting different CNN architectures or ratio based beauty justifications, however they are limited to pose constrains or smooth backgrounds. In our setting, an image can be uploaded from any scene with an applied filter or mask.

### **Proposed Method**

We solve this problem based on 3 steps. First, pre-train the network with Siamese layer [1][2] as enlarging the model by Net2Net [3] incrementally. Then fine-tune the model with Huber-Loss based regression for scoring and just before fine-tuning use Net2Net operator once more to double the model size.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/selfai_draw-1.png)

Method overview. 1. Train the model with Siamese layer, 2. Double the model size with Net2Net, 3. Fine-tune the model with Huber-Loss for scoring.

##### Siamese Network

Siamese network architecture is a way of learning which is embedding images into lower-dimensions based on similarity computed with features learned by a feature network. The feature network is the architecture we intend to fine-tune in this setting. Given two images, we feed into the feature network and compute corresponding feature vectors. The final layer computes pair-wise distance between computed features and final loss layer considers whether these two images are from the same class (label 1) or not (label -1) .

![Siammese network. From [2]](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/siamese.png)

Siamese network. From [2]. Both convolutional network shares parameters and learning the representation in parallel. In  our setting, these parameters belong to our network to be fine-tuned.

Suppose ![G_w()](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_9beca4d024a3f7b25c0e5eeea5c732a2.gif)G\_w() is the function implying the feature network and ![X](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_02129bb861061d1a052c592e2dc6b383.gif) is raw image pixels. Lower indices of ![X](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_02129bb861061d1a052c592e2dc6b383.gif) shows different images. Based on this parametrization the final layer computes the below distance (L1 norm).

![E_w = ||G_w(X_1) - G_W(X_2)||](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_a5f5ddcf595d9f5751b0083118601e7b.gif)E\_w = ||G\_w(X\_1) - G\_W(X\_2)||

On top of this any suitable loss function might be used. There are many different alternatives proposed lately. We choose to use Hinge Embedding Loss which is defined as,

![L(X, Y) = \begin{cases} x_i, & \text{if }\ y_i=1 \\ \text{max}(0, margin-x_i), & \text{if}y_i=-1 \end{cases} ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_ebafc635e8c5503cfe5fc2a6824b189a.gif)L(X, Y) = \begin{cases} x\_i, & \text{if }\ y\_i=1 \\ \text{max}(0, margin-x\_i), & \text{if}y\_i=-1 \end{cases}

Here in this framework, Siamese layer tries to push the network to learn features common for the same classes and differentiating for different classes..  Being said this, we expect to learn powerful features capturing finer details compared to simple supervised learning with help of the pair-wise consideration of examples. These features present good initialization for latter stage fine-tuning in relation to simple random or ImageNet initialization.

![Siamese network tries to contract instances belonging to the same classes and disperse instances from different classes in the feature space.](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/siamese2.png)

Siamese network tries to contract instances belonging to the same classes and disperse instances from different classes in the feature space.

##### Architecture update by Net2Net

Net2Net [3] proposes two different operators to make the networks deeper and wider while keeping the model activations the same. Hence, it enables to train a network incrementally from smaller and shallower to wider and deeper architectures. This accelerates the training, lowers computational requirements and results possibly better representations.

![](https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/16cb6876666f3a7b56a636c1d85ad00bd0d98bf3/1-Figure1-1.png)

Figure from Net2Net slide

We use Net2Net to reduce the training time in our modest computing facility and benefit from Siamese training without any architectural deficit. We apply Net2Net operators once in everytime training stalls through Siamese traning. In the end of the Siamese training we applied Net2Net wider operation once more to double the size and increase model capability to learn more representation.

Wider operation adds more units to a layer by copying weights from the old units and normalizes the next layer weights by the cloning factor of each unit, in order to keep the propagated activation the same.  Deeper operation adds an identity layer between successive layers so that again the propagated activation stands the same.

One subtle difference in our use of Net2Net is to apply zeroing noise to cloned weights in wider operation. It basically breaks the symmetry and forces each unit to learn similar but different representations.

**Sidenote:** I studied this exact method in parallel to this paper at Qualcomm Research when I was participating ImageNet challenge. However, I cannot find time to publish before Net2Net.  Sad 🙁

##### Fine-tuning

Fine-tuning is performed with Huber-Loss on top of the network which was used as the feature network at Siamese stage.  Huber-Loss is the choice due to its resiliency to outlier instances. Outliers are extremely harmful in fine-grain problems (miss-labeled  or corrupted instance) especially for small scale data sets. Hence, it is important for us to reconcile the effect of wrongly scored instances.

As we discussed above, before fine-tuning, we double the width (number of units in each layer) of the network. It enables to increase the representation power of the network which seems important for fine-grain problems.

##### Data Collection and Annotation

For this mission, we collect ~100.000 images from the web,  prune the irrelevant or low-quality images then annotate the remaining ones  on a crowd-sourced website. Each image is scored between 0 to 9.  Eventually, we have 30.000 images annotated where each one is scored at least twice by different annotators.

Understanding of beauty varies among cultures and we assume that variety of annotators minimized any cultural bias.

Annotated images are processed by face detection and alignment procedure in order to focus faces centered and aligned by the eyes.

##### **Implementation Details**

For all the model training,  we use Torch7 framework and almost all of the training code is released on [Github](https://github.com/erogol/resnet.torch) . In this repository, you find different architectures at different code branches.

Fine-tuning leverages a data sampling strategy alleviating the effect of data imbalance.  Our data set includes a a Gaussian like distribution over the classes in which mid-classes have more instances compared to fringes.  To alleviate this, we first pick a random class then select a random image belonging to that class. That gives equal change to each class to be selected.

We applied rotation, random scaling, color noise and random horizontal flip for data augmentation.

We do not use Batch Normalization (BN) layers since they lavish computational cost and in our experiments we obtain far worse performances. We believe it relies on the fine-detailed nature of the problem and BN layers just loose the representational power of the network due to implicit noise applied by its layers.

ELU activation is used for all our network architectures since, approving the claim of [8], it accelerates the training of a network without BN layers.

We tried many different architectures but with a simple and memory efficient model ([Tiny Darknet](http://pjreddie.com/darknet/tiny-darknet/))  was enough to obtain comparable performance in shorter training time. Below, I share Torch code for the model definition;

```python


-- The Tiny Model

model:add(Convolution(3,16,3,3,1,1,1,1))
model:add(ReLU())
model:add(Max(2,2,2,2,0,0))

model:add(Convolution(16,32,3,3,1,1,1,1))
model:add(ReLU())
model:add(Max(2,2,2,2,0,0))

model:add(Convolution(32,16,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(16,128,3,3,1,1,1,1))
model:add(ReLU())
model:add(Convolution(128,16,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(16,128,3,3,1,1,1,1))
model:add(ReLU())
model:add(Max(2,2,2,2,0,0))

model:add(Convolution(128,32,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(32,256,3,3,1,1,1,1))
model:add(ReLU())
model:add(Convolution(256,32,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(32,256,3,3,1,1,1,1))
model:add(ReLU())
model:add(Max(2,2,2,2,0,0))

model:add(Convolution(256,64,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(64,512,3,3,1,1,1,1))
model:add(ReLU())
model:add(Convolution(512,64,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(64,512,3,3,1,1,1,1))
model:add(ReLU())
model:add(Convolution(512,128,1,1,1,1,0,0))
model:add(ReLU())
model:add(Convolution(128,1024,3,3,1,1,1,1))
model:add(ReLU())
model:add(Avg(14,14,1,1))

model:add(nn.View(1024):setNumInputDims(3))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1024, 1))

```

### Experiments

In this section, we will discuss what are the contributions of individual bits and pieces of the proposed method. For any numerical comparison, I show correlation between the model prediction and the annotators score in a validation set.

##### Effect of Pre-Training

Pre-training with Siamese loss depicts very crucial effect. The initial representation learned by Siamese training presents a very effective initialization scheme for the final model.  Without pre-training, many of our train runs stall so quickly or even not reduce the loss.

Correlation values with different settings, higher is better;

* **with pre-training : 0.82**
* without pre-training : 0.68
* with ImageNet: 0.73

##### Effect of Net2Net

The most important aspect of Net2Net is to allow training incrementally, in a faster manner. It also reduces the engineering effort to your model architecture so that you can validate smaller version of your model  rapidly before training the real one.

In our experiments, It is observed that Net2Net provides good speed up. It also increase the final model performance slightly.

Correlation values with different settings;

* **pre-training + net2net :** **0.84**
* with pre-training : 0.82
* without pre-training : 0.68
* with ImageNet (VGG): 0.73

Training times;

* **pre-training + net2net :** **5 hours**
* with pre-training : 8 hours
* without pre-training : 13 hours
* with ImageNet (VGG): 3 hours

We can see the performance and time improvement above. Maybe 3 hours seems not crucial but think about replicating the same training again and again to find the best possible setting. In such case, it saves a lot.

### **Deficiencies**

Although, proposed method yields considerable performance gain, correcting the common notion, more data would increase the performance much beyond. It might be observed by the below learning curve that our model learns training data very-well but validation loss stalls quickly. Thus, we need much more coverage by the training data in order to generalize better on validation set.

![train_loss_curve](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/train_loss_curve.png)

Sample training curve from of the fine-tuning stage. Early saturation on validation loss is a sign of requirement for more training data.

In this work, we only consider simple and efficient model architectures. However, with more resources, more complex network architectures might be preferred and that might result additional gains.

We do not separate man and woman images since we believe that the model is supposed to learn genders implicitly and score accordingly. It is not experimented yet so such grouping likely to increase the performance.

### Visualization

Below we see a simple occlusion analysis of our network indicating the model's attention while scoring. This is done by occluding part of the image in sliding window fashion and compute absolute prediction changes in relation to normal image.

Figures show that, it mainly focuses on face and specifically eyes, nose and lips for high score images where as attention is more scattered for low and medium scale scores.  
![adriana_selfai](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/adriana_selfai.png)![pitt_selfai](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/pitt_selfai-1.png)![deep_selfai](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/deep_selfai.png)![doutzen_selfai](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/doutzen_selfai-1.png)![doutzen_selfai2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/doutzen_selfai2.png)

![ugly_selfai](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/ugly_selfai.png)

Model's attention based on occlusion sensitivity.

Below, we have random top and low scored selfies from validation set . It seems like results are not perfect but still its predictions are concordant to our inclination to these images.

![out](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/out.jpg)

Top scored images from validation set

![out](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/out-1.jpg)

bottom scored images from validations set.

### Conclusion

Here, we solidify the ability of deep learning models, CNNs in particular. Results are not perfect but still make sense and amaze me. It looks very intriguing that how couple of matrix multiplication is able to capture what is beautiful and what is not.

This work entails to [Selfai](http://selfai-app.8bit.ai) mobile application, you might like to give it a try for fun (if you did not before reading it). For instance, I stop growing my facial hair after I see a huge boost of my score. Thus it might be used as a smart mirror as well :). There is also the Instagram account where selfai bot scores images tagged #selfai\_robot or sent by direct message.

Besides all, keep in mind that this is just for fun without any bad intention. It was sparked by curiosity and resulted these applications.

Finally, please share your thoughts, comment and more. It is good to see what people think about your work.

**Disclaimer:**This post is just a draft of my work to share this interesting problem and our solution with the community . This work might be a paper with some more legitimate future work.

**References**

[1] J. Bromley, I. Guyon, Y. LeCun, E. Sackinger, and R. Shah. Signature verification using a siamese time delay neural network. J. Cowan and G. Tesauro (eds) Advances in Neural Information Processing Systems, 1993.

[2] Chopra, S., Hadsell, R., & LeCun, Y. (n.d.). Learning a Similarity Metric Discriminatively, with Application to Face Verification. 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), 1, 539–546. http://doi.org/10.1109/CVPR.2005.202

[3]Chen, T., Goodfellow, I., & Shlens, J. (2015). Net2Net: Accelerating Learning via Knowledge Transfer. arXiv Preprint, 1–10. Retrieved from http://arxiv.org/abs/1511.05641

[4]Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, 2016.

[5]Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Re- thinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[6]Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations, 1–14. http://doi.org/10.1016/j.infsof.2008.09.005

[7]Huang, G., Liu, Z., & Weinberger, K. Q. (2016). Densely Connected Convolutional Networks. arXiv Preprint, 1–12. Retrieved from http://arxiv.org/abs/1608.06993

[8]Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2015). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). Under Review of ICLR2016， 提出了ELU, (1997), 1–13. Retrieved from http://arxiv.org/pdf/1511.07289.pdf%5Cnhttp://arxiv.org/abs/1511.07289%5Cnhttp://arxiv.org/abs/1511.07289


### Related posts:

1. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
2. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
3. [Object Detection Literature](http://www.erogol.com/object-detection-literature/ "Object Detection Literature")
4. [Why do we need better word representations ?](http://www.erogol.com/need-better-word-representations/ "Why do we need better word representations ?")