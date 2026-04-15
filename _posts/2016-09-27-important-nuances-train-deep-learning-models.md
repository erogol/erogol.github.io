---
layout: post
title: "Important Nuances to Train Deep Learning Models."
description: "!datasplitting(https://cdn"
tags: deep learning reference talk training tricks
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

![datasplitting](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_519/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/09/datasplitting.png)

A crucial problem in a real DL system design is to capture test data distribution with the trained model which only sees the training data distribution.  Therefore, it is always important to find a good data splitting scheme which at least gives the right measures to such divergence.

It is always a waste to spend all your time for fine-tunning your model on the measure of validation data taken from training data only. Because, when you deploy the model, it undergoes new instances sampled from dynamically shifting data distribution. If you have a chance to see some samples from this dynamic environment, use that to test your model on these real instances and keep your model more coherent and don't mislead your training flow.

That being said, on the above figure, the second row depicts the right way to choose your data split. And the third row shows the smoothed version which is suggested in practice.

![problem_relation](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_456/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/09/problem_relation.png)

Above figure shows common machine learning problems in relation to different components of your work flow. It is really important to understand what is really said here and what these problems explain.

Bias is the quality of your model on training data. If it predicts wrong on training, it has a "Bias" problem. If you have a good performance on training data but not on validation data, it yields "Variance" problem. If performance differs for validation data taken from training set and test set, it is "Train - Test mismatch". If performance suffers due to distribution shift on test time, it is "Overfitting".

Bias requires better architecture and longer training. Variance needs more data and regularization. Train - Test mismatch needs more training data from distribution similar to your test data. Overfitting needs regularization, more data, and data synthesis effort.

![training_model](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_525/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/09/training_model.png)

Above chart shows a salient way of conducting  DL system evolution.  Follow these decisions with empirical evidences and don't skip any of these in order not to be disappointed in the end. (I said it with many disappointments 🙂 )

![human_limit](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_600/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/09/human_limit.png)

When we see that train, validation errors are close enough to human level performance, it means more variance problem and we need to  collect more data similar to test portion and hurdle more data synthesis work. Train and validation errors  far from human level performance is the sign of bias problem, requires larger models and more training time. Keep in mind that, human performance is not the limit of what your model is theoretically capable of.

**Disclaimer**: Figures are taken from https://kevinzakka.github.io/2016/09/26/applying-deep-learning/ which summarizes Andrew Ng's talk.

**EDIT:**

From NIPS 2016:

![](https://gist.github.com/stuhlmueller/d809bb7b4a9dc03bf75f695a0f3ea2e4/raw/e3a22a43a618bd95cdd8aa29b2ac342f224c7399/recipe-1.jpg)

![](https://gist.github.com/stuhlmueller/d809bb7b4a9dc03bf75f695a0f3ea2e4/raw/e3a22a43a618bd95cdd8aa29b2ac342f224c7399/recipe-2.jpg)

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.