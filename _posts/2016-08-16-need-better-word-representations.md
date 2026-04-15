---
layout: post
title: "Why do we need better word representations ?"
description: "A successful AI agent should communicate"
tags: deep learning machine learning nlp
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

A successful AI agent should communicate. It is all about language. It should understand and explain itself in words in order to communicate us.  All of these spark with the "meaning" of words which the atomic part of human-wise communication. This is one of the fundamental problems of Natural Language Processing (NLP).

"meaning" is described as "the idea that is represented by a word, phrase, etc. How about representing the meaning of a word in a computer. The first attempt is to use some kind of hardly curated taxonomies such as WordNet. However such hand made structures not flexible enough, need human labor to elaborate and  do not have semantic relations between words other then the carved rules. It is not what we expect from a real AI agent.

Then NLP research focused to use number vectors to symbolize words. The first use is to donate words with discrete (one-hot) representations. That is, if we assume a vocabulary with 1K words then we create a 1K length 0 vector with only one 1 representing the target word.

word = [0 0 0 0 0 0 0 0 1 0 0 0 0 ... ]

Still one-hot encoding is deficient to  capture the semantic relations since any distance between two vectors is zero unless they are the same.

Then the idea is extended to exploit the words surrounding the target word. The core assumption, which is valid to some degree, two words sharing the similar meaning are surrounded by the similar set of words like "dog" and "cat".

For this purpose, a cooccurence matrix is computed by looking the surrounding words in a given window size.

![An example to illustrate the cooccurence matrix ](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/cooccurance_matrix.png)

An example to illustrate the cooccurence matrix with window size 1.

As the example above, we use any row to represent the corresponding word, and similarity between each of these vectors would give a similarity measure.

However, cooccurence matrix is not really efficient. It is hard to update with new words. It is computationally not feasible for especially very large vocabularies which is the case for many real-life problems. The representation is very sparse and lengthy which is a problem for subsequent classification problems.

What we can do is to quantize the lengthy representations to compact vectors. The first idea is to use SVD like factorization methods over cooccurence matrices.

```python

import numpy as np
la = np.linalg
words = [...]
X = np.array([...]) #cooccurence matrix
U,s,V = la.svd(X, full_matrices=False)
# each column of U is presenting different dimension of the words.
```

When you plot on the first two columns of the U vectors;

![Example representation of words.](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/words_on_space.png)

Example representation of words.

The consequent representations are dense and more compact. However, one problem is still floating.  The complexity of SVD scales quadratically as we add more words to our vocabulary. The solution is to learn representations indirectly. This is the point where the latest word2vec and Glove algorithms take the stage.  They rely on the simple core idea of predicting surrounding words from the given word, instead of direct computation of cooccurence matrices. We are able to learn word presentations by simple SGD based optimization which ease the problem really well and yields state-of-art results for many benchmark problems.

These algorithms are also able to learn analogies between the words which is what we aimed at the first place. We start with very concrete word presentations to soft dense vectors which donate word "meaning" in a digitized environment.

![Example to word analogies learned by the representation.](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/word_analogies.png)

Example to word analogies learned by the representation.

The progress on word representations unveils the more advance research efforts like Q&A machines, machine authors and human quality document classifiers.

**Disclaimer**: All figures are taken from  Richard Socher's slide on Stanford cs224. This post is my memory as a beginner in NLP sector of Deep Learning research.


### Related posts:

1. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
2. [NegOut: Substitute for MaxOut units](http://www.erogol.com/negout-substitute-for-maxout-units-2/ "NegOut: Substitute for MaxOut units")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")