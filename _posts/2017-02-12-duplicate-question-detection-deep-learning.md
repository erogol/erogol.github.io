---
layout: post
title: "Duplicate Question Detection with Deep Learning on Quora Dataset"
description: "Quora recently announced the first public dataset(https://data"
tags: deep learning duplicate detection machine learning quora siamese network
minute: 7
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Quora recently announced the [first public dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) that they ever released. It includes 404351 question pairs with a label column indicating if they are duplicate or not.  In this post, I like to investigate this dataset and at least propose a baseline method with deep learning.

Beside the proposed method, it includes some examples showing how to use Pandas, Gensim, Spacy and Keras. For the full code you check [Github](https://github.com/erogol/QuoraDQBaseline).

## Data Quirks

There are 255045 negative (non-duplicate) and 149306 positive (duplicate) instances. This induces a class imbalance however when you consider the nature of the problem, it seems reasonable to keep the same data bias with your ML model since negative instances are more expectable in a real-life scenario.

When we analyze the data, the shortest question is 1 character long (which is stupid and useless for the task) and the longest question is 1169 character (which is a long, complicated love affair question). I see that if any of the pairs is shorter than 10 characters, they do not make sense thus, I remove such pairs.  The average length is 59 and std is 32.

There are two other columns "q1id" and "q2id" but I really do not know how they are useful since the same question used in different rows has different ids.

Some labels are not true, especially for the duplicate ones. In anyways, I decided to rely on the labels and defer pruning due to hard manual effort.

## Proposed Method

#### Converting Questions into Vectors

Here, I plan to use Word2Vec to convert each question into a semantic vector then I stack a Siamese network to detect if the pair is duplicate.

Word2Vec is a general term used for similar algorithms that embed words into a vector space with 300 dimensions in general.  These vectors capture semantics and even analogies between different words. The famous example is ;

> `king - man + woman = queen.`

Word2Vec vectors can be used for may useful applications. You can compute semantic word similarity, classify documents or input these vectors to Recurrent Neural Networks for more advance applications.

There are two well-known algorithms in this domain. One is Google's network architecture which learns representation by trying to predict surrounding words of a target word given certain window size. GLOVE is the another methos which relies on co-occurrence matrices. GLOVE is easy to train and it is flexible to add new words out-side of your vocabulary. You might like visit this [tutorial](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html) to learn more and check this [brilliant use-case](https://demos.explosion.ai/sense2vec/?word=fair%20game&sense=auto) Sense2Vec.

We still need a way to combine word vectors for singleton question representation. One simple alternative is taking the mean of all word vectors of each question. This is simple but really effective way for document classification and I expect it to work for this problem too.   In addition,  it is possible to enhance mean vector representation by using TF-IDF scores defined for each word. We apply weighted average of word vectors by using these scores. It emphasizes importance of discriminating words and avoid useless, frequent words which are shared by many questions.

#### Siamese Network

I described Siamese network in a previous [post](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/). In short, it is a two way network architecture which takes two inputs from the both side. It projects data into a space in which similar items are contracted and dissimilar ones are dispersed over the learned space. It is computationally efficient since networks are sharing parameters.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/11/siamese2.png)

Siamese network tries to contract instances belonging to the same classes and disperse instances from different classes in the feature space.

## Implementation

Let's load the training data first.

For this particular problem, I train my own GLOVE model by using [Gensim](https://radimrehurek.com/gensim/).

The above code trains a GLOVE model and saves it. It generates 300 dimensional vectors for words. Hyper parameters would be chosen better but it is just a baseline to see a initial performance. However, as I'll show this model gives performance below than my expectation. I believe, this is because our questions are short and does not induce a semantic structure that GLOVE is able to learn a salient model.

Due to the performance issue and the observation above, I decide to use a pre-trained GLOVE model which comes free with [Spacy](https://spacy.io/). It is trained on Wikipedia and therefore, it is stronger in terms of word semantics. This is how we use Spacy for this purpose.

Before going further, I really like Spacy. It is really fast and it does everything you need for NLP in a flash of time by hiding many intrinsic details. It deserves a good remuneration.  Similar to Gensim model, it also provides 300 dimensional embedding vectors.

The result I get from Spacy vectors is above Gensim model I trained. It is a better choice to go further with TF-IDF scoring.  For TF-IDF, I used [scikit-learn](http://scikit-learn.org/) (heaven of ML).  It provides TfIdfVectorizer which does everything you need.

After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores. The below code does this for just "question1" column.

Now, we are ready to create training data for Siamese network. Basically, I've just fetch the labels and covert mean word2vec vectors to numpy format. I split the data into train and test set too.

In this stage, we need to define Siamese network structure. I use [Keras](https://keras.io/) for its simplicity. Below, it is the whole script that I used for the definition of the model.

I share here the best performing network with residual connections. It is a 3 layers network using Euclidean distance as the measure of instance similarity. It has Batch Normalization per layer. It is particularly important since BN layers enhance the performance considerably. I believe, they are able to normalize the final feature vectors and Euclidean distance performances better in this normalized space.

I tried Cosine distance which is more concordant to Word2Vec vectors theoretically but cannot handle to obtain better results. I also tried to normalize data into unit variance or L2 norm but nothing gives better results than the original feature values.

Let's train the network with the prepared data. I used the same model and hyper-parameters for all configurations. It is always possible to optimize these but hitherto I am able to give promising baseline results.

## Results

In this section, I like to share test set accuracy values obtained by different model and feature extraction settings.  We expect to see improvement over 0.63 since when we set all the labels as 0, it is the accuracy we get.

These are the best results I obtain with varying GLOVE models. they all use the same network and hyper-parameters after I find the best on the last configuration depicted below.

* Gensim (my model) + Siamese: 0.69
* Spacy + Siamese :  0.72
* Spacy + TD-IDF + Siamese : **0.79**

We can also investigate the effect of different model architectures.  These are the values following  the best word2vec model shown above.

* 2 layers net : 0.67
* 3 layers net + adam : 0.74
* 3 layers resnet (after relu BN) + adam : 0.77
* 3 layers resnet (before relu BN) + adam : 0.78
* 3 layers resnet (before relu BN) + adam + dropout : 0.75
* 3 layers resnet (before relu BN) + adam + layer concat : **0.79**
* 3 layers resnet (before relu BN) + adam + unit\_norm + cosine\_distance : Fail

Adam works quite well for this problem compared to SGD with learning rate scheduling. Batch Normalization also yields a good improvement. I tried to introduce Dropout between layers in different orders (before ReLU, after BN etc.), the best I obtain is 0.75.  Concatenation of different layers improves the performance by 1 percent as the final gain.

In conclusion, here I tried to present a solution to this unique problem by composing different aspects of deep learning. We start with Word2Vec and combine it  with TF-IDF and then use Siamese network to find duplicates. Results are not perfect and akin to different optimizations. However, it is just a small try to see the power of deep learning in this domain. I hope you find it useful :).

## Updates

* Switching last layer to FC layer improves performance to 0.84.
* By using bidirectional RNN and 1D convolutional layers together as feature extractors improves performance to 0.91. Maybe I'll explain details with another post.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [NegOut: Substitute for MaxOut units](http://www.erogol.com/negout-substitute-for-maxout-units-2/ "NegOut: Substitute for MaxOut units")
2. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
3. [Why do we need better word representations ?](http://www.erogol.com/need-better-word-representations/ "Why do we need better word representations ?")
4. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")