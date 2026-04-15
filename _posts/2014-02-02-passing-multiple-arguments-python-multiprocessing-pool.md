---
layout: post
title: "Passing multiple arguments for Python multiprocessing.pool"
description: "Python is a very bright language that is used by variety of users and mitigates many of pain"
tags: code multiprocessing parallel-computing python
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Python is a very bright language that is used by variety of users and mitigates many of pain.

One of the core functionality of Python that I frequently use is [multiprocessing](http://docs.python.org/2/library/multiprocessing.html) module. It is very efficient way of distribute your computation  embarrassingly.

If you read about the module and got used, at some point you will realize, there is no way proposed to pass multiple arguments to parallelized function. Now, I will present a way to achieve in a very Pythonized way.

For our instance, we have two lists with same number of arguments but they need to be fed into the function which is pooling.

Here we have self cover code:


### Related posts:

1. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
2. [Simple Parallel Processing in Python](http://www.erogol.com/simple-parallel-processing-python/ "Simple Parallel Processing in Python")
3. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
4. [Fundamental Sort Algorithms in Python](http://www.erogol.com/fundamental-sort-algorithms-python/ "Fundamental Sort Algorithms in Python")