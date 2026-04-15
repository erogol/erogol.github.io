---
layout: post
title: "How to use Python Decorators"
description: "Decorators are handy sugars for Python programmers to shorten things and provides more concise progr"
tags: codebook coding python
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Decorators are handy sugars for Python programmers to shorten things and provides more concise programming.

For instance you can use decorators for user authentication for your REST API servers. Assume that, you need to auth. the user for before each REST calls. Instead of appending the same procedure to each call function, it is better to define decorator and tagging it onto your call functions.

Let's see the small example below. I hope it is self-descriptive.

```python


"""
How to use Decorators:

Decorators are functions called by annotations
Annotations are the tags prefixed by @
"""

### Decorator functions ###
def helloSpace(target_func):
def new_func():
print "Hello Space!"
target_func()
return new_func

def helloCosmos(target_func):
def  new_func():
print "Hello Cosmos!"
target_func()
return new_func


@helloCosmos # annotation
@helloSpace # annotation
def hello():
print "Hello World!"

### Above code is equivalent to these lines
# hello = helloSpace(hello)
# hello = helloCosmos(hello)

### Let's Try
hello()

```


### Related posts:

1. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
2. [Simple Parallel Processing in Python](http://www.erogol.com/simple-parallel-processing-python/ "Simple Parallel Processing in Python")
3. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
4. [Some inefficient algorithms you don't need to know!](http://www.erogol.com/inefficient-algorithms-dont-need-know/ "Some inefficient algorithms you don't need to know!")