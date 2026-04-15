---
layout: post
title: "Comparison of Deep Learning Libraries After Years of Use"
description: "As we witness the golden age of AI and deep learning, there are many different tools and frameworks "
tags: caffe deep learning keras lasagne machine learning
minute: 13
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

As we witness the golden age of AI and deep learning, there are many different tools and frameworks continuously proposed by different communities. Sometimes it is even hard to catch up what is going on. You choose one over another then you see a new library and you go for it. However, it seems the exact choice is not obvious to anyone.

From my point of view, libraries are measured by flexibility and run-time trade-off. If you go with a library which is really easy to use, it is slow as much as that. If the library is so fast, then it does not serve that mush of flexibility or it is so specialized to a particular type of models like Convolutional NNs, hence they do not support the type of your interest such as Recurrent NNs.

After all the tear, shed and blood dropped by years of experience in deep learning, I decide to share my own intuition and opinion about the common deep learning libraries so that these might help you to choose the right one for your own sake .

Let's start by defining some metrics to evaluate a library. These are the pinpoints that I consider;

1. **Community support :**  It is really important, especially for a beginner to ask questions and gather answers to learn the library. This is basically related to success and visibility of the community of the library.
2. **Documentation:** Even you are familiar to a library, due to their extensive and evolving nature , updated documentation is really vital for a user.  A library might be the best but, it also need a solid documentation to prove it to a user.
3. **Stability:**Many of the libraries are open-source. It is of course good to have the all but open-source means more fragile and buggy implementations. It is also really hard to understand in advance that a library is stable enough to use for your code. It needs time to investigate and the worse is to see the real problem at the end of your implementation. It is really disruptive, I experienced once and never again 🙂
4. **Run-time performance:** It includes; GPU, CPU run-times and use of the hardware capabilities, distributed training with multiple-GPUs on single machine and multiple machines and memory use which limits the models you train.
5. **Flexibility:** Experimenting new things and development of new custom tools and layers are also crucial part of the game. If you are a researcher it is maybe the foremost point that you count on.
6. **Development:**Some libraries are being developed with a great pace and therefore it is always easy to find new functionalities and state-of-art layers and functions. It is good from that point but sometimes it makes the library hard to consolidate, if especially it has deficient documentation resources.
7. **Pre-trained models and examples:**For a hacker to use deep learning, this is the most important metric. Many of the successful deep learning models are trained by using big computer clusters with very extensive experimentations. Not every one is able to budget up for such computation power. Therefore, it is important to have pre-trained models to step into.

Below with each heading of library, I discuss the library on these points.

### Torch

[Torch](http://torch.ch/) is the one which I use as the latest. As most of you know,  Torch is a Lua based library and used extensively by Facebook and Twitter Research teams for deep learning products and research.

1. **Community Support:**It has a relatively small community compared to other libraries but still it is very responsive to any problem and question that you encounter.
2. **Documentation:** [Good documentation](http://torch.ch/docs/package-docs.html) is waiting for you. But still if you are new to Lua too, sometimes it is not enough and it leaves you to google more.
3. **Stability:** It is really stable. I couldn't see any problem in terms of robustness yet.
4. **Run-time performance:**It is the most powerful metric of Torch. It uses all the capability of any hardware you use. You can switch different hardware supports by importing regarding modules. It is not that invisible but still easy to convert you CUDA code to CPU or vice a versa.  One another difference, you need to convert your CPU or GPU model to another architecture to use change its basis. A trained model is not compatible with all architectures without a touch. Thus you see many questions asking "How can I convert GPU model to CPU?". It is not hard but you need to know.  It is also very easy to use multiple-GPUs in single machine but yet I do not see any support for distributed training in multi-machine setting.
5. **Flexibility:**Due to the weirdness of Lua, it is not my choice, if I need to develop something custom. Also it has no powerful auto-differentiation mechanism (AFAIK) so it needs you code the back-propagation function for your layers as well.  Despite of such caveats, it is still the most accessible library by the research people and publications.
6. **Development:**It is maybe the most successful library to follow up what is new in the deep learning literature. It is actively developed but it is not that open to third-party developers to contribute. At least this is my intuition.
7. **Pre-trained models and examples:**It has a pretty good pre-trained model support, in generaled released by facebook team such as [ResNet](https://github.com/facebook/fb.resnet.torch). Other than this, it supports to convert Caffe models like some other libraries. In addition, you can find different examples about different architectures or problems including NLP, Vision or some others.

**Summary**

I really like the performance and its aggressive resource use on CPU or GPU. However, I observe a bit more memory use in GPU which is a bottleneck for training large models. I'd personally use Torch for my main tool but Lua seems still very intricate compared to Python. For a guy ,like me, who uses Python for everything, using Torch models complicates the development. Still there is a great library [Lutorpy](https://github.com/imodpasteur/lutorpy) which makes Torch model plausible from Python.

### MxNet

[MxNet](http://mxnet.readthedocs.io/en/latest/system/index.html) is a library backed by [Distribute Machine Learning Community](https://github.com/dmlc) that already conducted many great project such as the dream tool [xgboost](https://github.com/dmlc/xgboost) of many Kagglers . Albeit it is not highlighted on web or deep learning communities,  it is really powerful library supporting many different languages; Python, Scala, R.

1. **Community Support:**Every conversation and question is going on through github issues and many problems are answered directly by the core developers which is great. Beside, you need sometime to gather your answers.
2. **Documentation:**Compared to its really fast development progress, its documentation falls slightly behind. It is better to follow merge requests on the repo and read raw codes to see what is happening. Beside of that, it has very good collection of examples and tutorials waiting in different formats and languages.
3. **Stability:** Dense development effort causes some instability issues. I experienced couple of those .  For instance, a trained model gives different outputs with different back-end architectures. I guess they solved it to some extend but still I see it with some of my models.
4. **Run-time performance:**I believe, it is the fastest training time library. One important note is to set all the options well for your machine configuration. It has really efficient memory footprint compared to other libraries by its [optimized computational graph paradigm](http://mxnet.readthedocs.io/en/latest/system/index.html). I am able to train many large models that are not allowed by the other libraries. It is very easy to use multiple GPUs and it supports distributed training as well by distributed SGD algorithm.
5. **Flexibility:**It is based on a third party Tensor computation library developed in C++, called MShadow. Therefore, you need to learn that first to develop custom things utilizing full potential of the library. You are also welcome to code custom things through language interfaces like Python. It is also possible to use implemented blocks to create some custom functionalities as well. However, to be honest, I did not see so much researcher using MxNet.
6. **Development:**It inhibits really good development effort, mainly regulated by the core developers of the team. Still they're open to pull requests and discuss something new with you.
7. **Pre-trained models and examples:**You can convert some set of pre-trained Caffe models like VGG by the provided script. They also released InveptionV3 type of ImageNet networks and InceptionV2 type model trained on 21K ImageNet collection which is really great for fine-tuning. I also wait for ResNet but still none.

**Summary**

This is the library of my choice for many of my projects, mostly due to run-time efficiency, really solid Python support and less GPU memory use.

Some critics, MxNet mostly support Vision problems and they partially start to work on NLP architectures. You need to convert all data to their data format for the best efficiency, it slows the implementation time but makes things more efficient in terms of memory and hard-drive use. Still for small projects, it is a pain. You can convert your data to numpy array and use it but then you are not able to use extensive set of data augmentation techniques provided by the library.

### Theano

[This library](https://github.com/Theano/Theano) is maintained by Montreal group. It is the first of its kind as far as I know. It is a Python library which takes your written code and compiles it to C++ and CUDA. Hence, it targets machine learning applications, not just deep learning. It also converts the code to computational graph like MxNet then optimizes memory and execution. However, all these optimizations take a good time which is the real problem of the library. Since Theano is a general use machine learning library, following facts are based on deep learning libraries [Lasagne](http://lasagne.readthedocs.io/en/latest/index.html) and [Keras](https://github.com/fchollet/keras/tree/master/examples) which share many properties.

1. **Community Support:** They have both big communities supporting google user groups and github issue pages. I'd say Keras has more support then Lasagne. You can get any question answered quickly.
2. **Documentation:**Simple but powerful documentation for both. Once you got the logic behind these libraries, it is so fluid to develop your own models and applications. Each important subject is explained by a example which I really like to see from scikit-learn as well.
3. **Stability: T**hey are really high paced libraries. Due to Theano's simplicity to develop new things, they follow what is new easily but it is also dangerous in terms of stability. As far as you do not rely on these latest features, they are stable.
4. **Run-time performance:**They are bounded by the abilities of Theano and beside this any Theano based library just diverges by the programming techniques and the correct use of Theano codes.  The real problem for these libraries is the compile time in which you wait before model execution.  It is sometimes too much to bare, specially for large models. If you compile successfully, after the last update of Theano, it is really fast for training in GPU. I've not experienced CPU execution too much. Memory use is not that efficient compared to MxNet but still comparable with Torch. AFAIK, they started to support multi GPU execution after the last version of Theano but distributed training is still out of the scope.
5. **Flexibility:**Due to auto-differentiation of Theano and the syntactic goods of Python, it is really easy to develop something new. You only need to take a already implemented layer or a function then modify it to your custom thing.
6. **Development:** These libraries are really community driven open-source counterparts. They are so fast to capture what is new . Due to the easiness of development, sometimes one thing might have lots of alternative implementations.
7. **Pre-trained models and examples:** They provide VGG networks and there are scripts to convert Caffe models. However, I've not experimented converted Caffe models with these libraries.

**Summary**

If we need to compare Keras and Lasagne, Keras is more modular and hides all the details from the developer which reminds scikit-learn. Lasagne is more like a toolbox which you use to come up with more custom things.

I believe, these libraries are perfect for quick prototyping. Anything can be implemented in a flash of time without keeping the details out of your view.

### Caffe

[Caffe](https://github.com/BVLC/caffe) is the flagship of deep learning libraries for both industry and research. It is the first successful open-source implementation with very solid but simple foundation. You do not need to know code to use Caffe. You define your network with a description files and train it.

1. **Community Support:** It has maybe the largest community. I believe anyone interested in deep learning would have some experience with it.  It has a large and old google users group and github issues pages that are full of information.
2. **Documentation:** I always see that documentation is always a bit old compared to the current stage of the library. Even they do not have a extensive documentation page comparable to other libraries, you can always find tutorials and examples on web to learn more. A simple google query would give many different resources as well.
3. **Stability:**It is really solid library. It uses well-known libraries for matrix operations and CUDA calls. I've not seen any problem yet.
4. **Run-time performance:**It is not the best but always acceptable. It uses well-founded libraries for any run-time crucial operations like convolution. It is bounded by these libraries. Custom solutions are akin to better run-times but they also degrade the stability as that amount. You can switch to CPU or GPU backend by a simple call without any change of your code.  It does well in terms of memory consumption but still too much compared to MxNet especially Inception type models. One problem is that, it does not support GPUs other than Nvidia. There are of course branches but I've never used them. It supports multi-gpu training on single machine but not distributed training.
5. **Flexibility:**Learning to code with Caffe is not that hard but documentation is not helpful enough. You need to look to source code to understand what is happening and use present implementations to template your custom code. After you understand the basics, it is easy to use and bend the library as you need. It has a good interface to Python and is compatible to new layers written with Python. It is a good library which hides the GPU and CPU integration from the developer. Caffe is very acceptable by the research community as well.
6. **Development:**It has very broad developer support and many forks that target different applications but the master branch is so picky to something new. This is good for a stable library but also causes this many forks. For instance, Batch Normalization is merged with the master branch after years of wait and discussion.
7. **Pre-trained models and examples:**[Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) is the heaven of pre-trained models for variety of domains and the collection keeps increasing. It has good set of example codes that can initiate you own project.

**Summary**

Caffe is the first successful deep learning library from many different aspects. It is stable, efficient.

Sometimes it is a huge bother to define large models by a model description file. It makes things very wobbling and akin to be mistaken. For example, you can miss a number of mistype it then your model crushes. finding such small problems over hundreds of lines is a huge bother.  In such cases, Python interface is wiser choice by defining some functions to create common layers.

**NOTE:**This is all my own experience with these libraries. Please correct me if you see something wrong or deceitful. Hope this helps to you. BEST 🙂

### 



### Related posts:

1. [What is special about rectifier neural units used in NN learning?](http://www.erogol.com/what-is-special-about-rectifier-neural-units-used-in-nn-learning/ "What is special about rectifier neural units used in NN learning?")
2. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
3. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
4. [Kaggle Plankton Challenge Winner's Approach](http://www.erogol.com/kaggle-plankton-challenge-winners-approach/ "Kaggle Plankton Challenge Winner's Approach")