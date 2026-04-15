---
layout: post
title: "Short guide to deploy Machine Learning"
description: "!(https://cdn0"
tags: deep learning machine learning model selection model training
minute: 5
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

![](https://cdn0.vox-cdn.com/thumbor/8n1DnMVgIHQmo-mVliLK07kzVNY=/cdn0.vox-cdn.com/uploads/chorus_asset/file/3628610/a_simple_task_headline.0.jpg)

"Calling ML intricately simple 🙂 "

Suppose you have a problem that you like to tackle with machine learning and use the resulting system in a real-life project.  I like to share my simple pathway for such purpose, in order to provide a basic guide to beginners and keep these things as a reminder to myself. These rules are tricky since even-thought they are simple, it is not that trivial to remember all and suppress your instinct which likes to see a running model as soon as possible.

When we confronted any problem, initially we have numerous learning algorithms, many bytes or gigabytes of data and already established knowledge to apply some of these models to particular problems.  With all these in mind, we follow a three stages procedure;

1. Define a goal based on a metric
2. Build the system
3. Refine the system with more data

Let's pear down these steps into more details ;

### DEFINE GOAL & METRIC

##### Human Level vs Acceptable

First thing,  we need to adjust what is the expected quality from the system performance. We might expect human level performance if it is medical diagnoses system or we might prefer to have lower one, if it is a simple mobile application. This decision defines the cost (time, money and engineering) of the system. As we increase the our expectation, we also need to invest more.

##### What Metric to Measure

Thus don't go dinosaur hunt with your flip-flops.  Related to the problem at hand, define a right metric to gauge system performance. It's supposed to match with the nature of the problem. Possible alternatives are these;

* Accuracy  -  object classification
* Recall         -  medical diagnose
* Amount of error   -   rental price prediction for houses
* F-score      -  document classification

Defining the right metric creates a huge quality difference. It involves a process that you understand the user (or customer) well and find the matching criteria which apes the selective procedure of your user well in the artificial environment in which you develop your solution.

### BUILDING THE SYSTEM

##### Create a baseline ASAP

Do not try to devise the time machine without a clock. First devise a minimum viable system with any tool and algorithm, easy to use and implement. Define this as a baseline. Baseline is useful to show what is your gain, whether it is significant, random or what.

##### Improving Baseline

After you are done with the baseline system then you can start to add on. Here, following a incremental proceeding is an efficient strategy which makes things easier to follow. Then it is also easier to backup things against if there is something not working as expected.

Do not waste time with the state-of-art space level techniques. Let Occam speaks. Only go for more advance methods if the data demands so.

For instance, it is not always the right choice to use ImageNet winner Inception network directly to your problem. Define your model structure based on observations on your data. In general, if there is noise and data is easily separable then use shallower models. As noise decreases and structure increases in the data go for deeper and wider models.

The distinction between deeper and wider models is; deeper models are better to capture more high-level abstractions that are important to differentiate particularly different classes (car vs horse) and wider models are better for fine-grain problems where the classes are close to each other and only slight commonalities differentiate one from another (genre of cats).

##### Kind of Model

Based on your problem, there are better subset of ML models. These models might be used over each other but the best is always to keep the model nature and the problem nature aligned.

* Raw data  --> Fully connected network (MLP)
* Spatial data (Image) --> Convolutional network
* Temporal, sequential data --> Recurrent networks (LSTM, RNN, GRU)

### REFINE with DATA

Assume that you finalized your system with good success and you deployed it. It is not the end yet.  There are still work to be done.

##### Don't Believe Numbers

Until that point you always measure the success based on the metric values in a controlled environment. However,  these values might not be the indicators of the real-life. Data might change or your users might change. Thus, always check the system performance live after initial deployment. Do A/B test, check your metric values on real-time data, validate your hypothesis with real values.

##### Update with New Data

If you are able to obtain more data in time always use it to update your model and fine-tune it. It is the rule of thump that more data always increases the performance. Do not skip that since you might even achieve unimaginable results as you update the system with more and more data. This is also the skill of big ML driven companies like Google.  They are really skillful to use running data to enhance their products.

### Last Words

In this post there are many things I skipped such as details of training a model,  finding its defects and re-iterating to increase the performance.  You might like to see [my one another post](http://www.erogol.com/important-nuances-train-deep-learning-models/) to see such details.

Best luck with your next ML heist 🙂

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [A Large set of Machine Learning Resources for Beginners to Mavens](http://www.erogol.com/large-set-machine-learning-resources-beginners-mavens/ "A Large set of Machine Learning Resources for Beginners to Mavens")
2. [ML Work-Flow (Part 3) - Feature Extraction](http://www.erogol.com/ml-work-flow-part-3-feature-extraction/ "ML Work-Flow (Part 3) - Feature Extraction")
3. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
4. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")