---
layout: post
title: "Some Useful Machine Learning Libraries."
description: "Especially, with the advent of many different and intricate Machine Learning algorithms, it is very "
tags: libraries machine learning tools
minute: 8
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Especially, with the advent of many different and intricate Machine Learning algorithms, it is very hard to come up with your code to any problem. Therefore, the use of a library and its choice is imperative provision before you start the project. However, there are many different libraries having different quirks and rigs in different languages, even in multiple languages so that choice is not very straight forward as it seems.

Before you start, I strongly recommend you to experiment the library of your interest so as not to say " Ohh Buda!" at the end. For being a simple guide, I will point some possible libraries and signify some of them as my choices with the reason behind.

### **My simple bundle for small projects ----**

I basically use Python for my problems, in general. Here are my frequently used libraries.

* **[Scikit-learn](http://scikit-learn.org/stable/)** - Very broad and well established library. It has different functionalities that meet your requisites at your work flow. If you do not need some peculiar algorithms, Scikit-learn is just enough for all. It is predicated with Numpy and Scipy at Python. It also proposes very easy way to paralleling your code with very easy way.
* **[Pandas](http://pandas.pydata.org/)** - Other than being a machine learning library pandas is a "Data Analysis Library". It gives very handy features to have some observations on data, just before you design your work flow. It support in memory  and storage functions. Hence, It is especially useful, if your data is up to some large scales that is not easy to be handled via simple methods or cannot be fit into memory as a whole.
* **[Theano](http://deeplearning.net/software/theano/)** -  It is yet another Python library but it is a nonesuch library. Simply, it interfaces your python code to low-level languages. As you type in python like you do Numpy, it converts your code into prescribe low level counterparts and then compile them at that level. It gives very significant performance gains, particularly for large matrix operations. It is also able to utilize from GPU after simple configuration of the library without any further code change. One caveat is, it is not easy to debug  because of that compilation layer.
* **[NLTK](http://www.nltk.org/) -** It is a natural language processing tool with very unique and salient features. It also includes some basic classifiers like Naive Bayes. If your work is about text processing this is the right tool to process data.

### **Other Libraries -- (This list is being constantly updated.)**

##### **Deep Learning Libraries**

* **[Pylearn2](https://github.com/lisa-lab/pylearn2)** - "A machine learning research library". It is widely used especially among deep learning researches. It also includes some other features like Latent Dirichlet Allocation based on Gibbs sampling.
* **[Theanets](https://github.com/lmjohns3/theanets) (new) -**This is yet another Neural Networks library based on Theano. It is very simple to use and I think one of the best library for quick prototyping new ideas.
* **[Hebel](https://github.com/hannes-brt/hebel)**  - Another young alternative for Deep Learning implementation. "Hebel is a library for deep learning with neural networks in Python using GPU acceleration with CUDA through PyCUDA."
* **[Caffe](http://caffe.berkeleyvision.org/)** - A Convolutional Neural Network library for large scale tackles. It differs by having its own implemntation of CNN in low level C++ instead of well-known ImageNet implementation of Alex Krizhevsky. It assets faster alternative to Alex's code. It also provides MATLAB and Python interfaces.
* **[cxxnet](https://github.com/dmlc/cxxnet) -**Very similar to Caffe. It supports multi-GPU training as well. I've not used it extensively but it seems promising after my small experiments with MNIST dataset. It also servers very modular and easy development interface for new ideas. It has Python and Matlab interfaces as well.
* **[mxnet](https://github.com/dmlc/mxnet) -** This is a library from the same developers of cxxnet. It has additional features after the experience gathered from cxxnet and other backhand libraries. Different than cxxnet, it has a good interface with Python which provides exclusive development features for deep learning and even general purpose algorithms requiring GPU parallelism.
* **[Pybrain](http://pybrain.org/)** - "PyBrain is short for Python-Based Reinforcement Learning, Artificial Intelligence and Neural Network Library."
* **[Brainstorm](https://github.com/IDSIA/brainstorm)** - Python based, GPU possible deep learning library released by IDSIA lab. It is at ery early stage of development but it is still eye catching. At least for now, it targets recurrent networks and 2D convolution layers.

##### Linear Model and SVM Libraries

* **[LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/)** - A Library for Large Linear Classification. It is also interfaced by Scikit-learn.
* **[LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)** - State of art SVM library with kernel support. It has also third-party plug-ins, if its built-in capabilities are not enough for you.
* **[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki)** - I hear the name very often but haven't use it by now. However, it seems a decent library for fast machine learning.

##### **General Purpose Libraries**

* [**Shougun**](http://shogun-toolbox.org/) - General usage ML library, similar to Scikit-learn. It supports for different programming languages.
* [**MLPACK**](http://www.mlpack.org/) - "a scalable c++ machine learning library".
* [**Orange**](http://orange.biolab.si/) - One another general use ML library. "Open source data visualization and analysis for novice and experts". It has Self-Organizing ( I am studying on 🙂 ) maps implementation that diverse it from others.
* [**MILK**](http://packages.python.org/milk/) - "SVMs (based on libsvm), k-NN, random forests, decision trees. It also performs feature selection. These classifiers can be joined in many ways to form different classification systems."
* **[Weka](http://www.cs.waikato.ac.nz/ml/weka/)** - Weka is a very command tool for machine learning with GUI support. If you do not want to code, you can cull the data to Weka and select your algorithm from drop-menu, set the parameters and go. Moreover, you can call its functions from your java code. It supports some other languages as well.
* [**KNIME**](http://www.knime.org/) - Albeit I am not very fan of those kind of tools, KNIME is another example of GUI based framework. You just define your work-flow by creating a visual work-flow. Carry some process boxes to workspace, connect them as you want, set parameters and run.
* [**Rapid-Miner**](http://rapidminer.com/) - Yer another GUI based tool. It is very similar to KNIME but out of my practice, it has wider capabilities suited different domain of expertise.

##### Others

* **[MontePython](http://montepython.sourceforge.net/) -** Monte (python) is a Python framework for building gradient based learning machines, like neural networks, conditional random fields, logistic regression, etc. Monte contains modules (that hold *parameters*, a *cost-function* and a *gradient-function*) and trainers (that can adapt a module's parameters by minimizing its cost-function on training data).
* **[Modular Toolkit for Data Processing](http://mdp-toolkit.sourceforge.net/index.html)** - From the user’s perspective, MDP is a collection of supervised and unsupervised learning algorithms and other data processing units that can be combined into data processing sequences and more complex feed-forward network architectures.
* **[Statsmodels](http://statsmodels.sourceforge.net/)**is another great library which focuses on statistical models and is used mainly for predictive and exploratory analysis. If you want to fit linear models, do statistical analysis, maybe a bit of predictive modeling, then Statsmodels is a great fit.
* **[PyMVPA](http://www.pymvpa.org/index.html)** is another statistical learning library which is similar to Scikit-learn in terms of its API. It has cross-validation and diagnostic tools as well, but it is not as comprehensive as Scikit-learn.
* **[PyMC](http://pymc-devs.github.io/pymc/)** is the tool of choice for **Bayesians**. It includes Bayesian models, statistical distributions and diagnostic tools for the convergence of models. It includes some hierarchical models as well. If you want to do Bayesian Analysis, you should check it out.
* [**Gensim**](http://radimrehurek.com/gensim/) is topic modelling tool that is centered on Latent Dirichlet Allocation model. It also serves some degree of NLP functionalities.
* [**Pattern**](https://github.com/clips/pattern)-  Pattern is a web mining module for Python
* **[Mirado](http://fathom.info/mirador/)**-  is data visualization tool for complicated datasets supporting MAC and Win
* **[XGBoost (new)](https://github.com/tqchen/xgboost)** -  If you like Gradient Boosting models and you like to o it faster and stronger, it is very useful library with C++ backend and Python, R wrappers. I should say that it is far faster than Sklearn's implementation

### **My computation stack ---**

After the libraries, I feel the need of saying something about the computation environment that I use.

* **[Numpy](http://www.numpy.org/)[, Scipy,](http://www.scipy.org/) [Ipython](http://ipython.org/)[,](http://www.scipy.org/) [Ipython-Notebook](http://ipython.org/notebook.html)[,](http://www.scipy.org/) [Spyder](http://pythonhosted.org/spyder/) -** After waste some time with Matlab, I discovered those tools that  empower scientific computing with sufficient results. Numpy and Scipy are the very well-known scientific computing libraries. Ipython is an alternative to native python interpreter with very useful features. Ipython-Notebook is a very peculiar editor that is able to run on web-browser so it is good especially if you are working on a remote machine. Spyder is a python IDE and it has very useful capabilities that makes your experience very similar to Matlab. Last bu not least, all of them are very free. I really suggest to look at those items before you select a framework for your scientific effort.

At the end, for being self promoting I list my own ML codes ----

* **[KLP\_KMEANS](https://github.com/erogol/KLP_KMEANS)**- this is a very fast clustering procedure underpinned by Kohonen's Learning Procedure. It includes two alternative with basic Numpy and faster at large data Theano implementations.
* **[Random Forests](https://github.com/erogol/Random_Forests)** - It is Matlab code based on C++ back-end.
* [**Dominant Set Clustering**](https://github.com/erogol/DominantSetClustering) -  A Matlab code implementing very fast graph based clustering formulated by Replicator Dynamics Optimization.

W**Wikipedia:** W (named


### Related posts:

1. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
2. [Simple Parallel Processing in Python](http://www.erogol.com/simple-parallel-processing-python/ "Simple Parallel Processing in Python")
3. [Scikit based Machine Learning work-flow](http://www.erogol.com/scikit-based-machine-learning-work-flow/ "Scikit based Machine Learning work-flow")
4. [Paper review: ALL YOU NEED IS A GOOD INIT](http://www.erogol.com/need-good-init/ "Paper review: ALL YOU NEED IS A GOOD INIT")