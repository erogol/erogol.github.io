---
layout: post
title: "devil in the implementation details"
description: "I was hassling with interesting problem lately"
tags: research notes tips & tricks what i learn today
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I was hassling with interesting problem lately. I trained a custom deep neural network model with ImageNet and ended up very good results at least on training logs.  I used Caffe for all these. Then, I ported my model to python interface and give some objects to it. Boommm!not working and even raised random prob values like it is not even trained for 4 days. It was really frustrating. After a dozens of hours I discovered that "Devil is in the details" .

I was using one of the **Batch Normalization ("what is it ? "little intro [here](http://www.erogol.com/recent-advances-in-deep-learning/) )** PR that is not merged to master branch but seems fine.  Then I found that interesting problem. The code in the branch computes each batch's mean by only looking at that batch. When we give only one example at test time, then the mean values are exactly the values of this particular image. This disables everything and the net starts to behave strangely. After a small search I found the solution which uses moving average instead of exact batch average. Now, I am at the stage of implementation. The puchcard is, do not use any PR which is not merged to master branch, that simple 🙂

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.