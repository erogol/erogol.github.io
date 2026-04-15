---
layout: post
title: "Process of defining a machine learning solution (ML#2)"
description: "First of all we need to see How a ML algorithm is working"
tags: algorithm machine learning tutorial what is
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

First of all we need to see How a ML algorithm is working. Here is the schema.

[![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/08/MLProcess.jpeg "MLProcess")](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2012/08/MLProcess.jpeg)

A ML process takes these steps,

* Take the **training set .**
* **Train** your ML system with training set by the algorithm you select.
* Get an **hypothesis function** after all the training period.
* Get your next instances and **estimate next output**.

After knowing how ML process work, we need to aware of the process of defining a ML solution to a problem involves these steps.

* Interpret and think about efficient **representation** of your instances. For example define a instance as (xi,yi) where i means the i the instance of your set.
* **Define the measurement** of success for your machine learning solution. What is your performance measure. Is this the accuracy or what?
* **Define a Hypothesis** function representation. Do you want to get a linear function to separate different classes or more sophisticated ones.
* **Define the algorithm you use.** According to your pick of hypo. function you need to select your algorithm that might be useful to get such a hypo. function.
* **Implement** your algorithm. In addition most ML professional choose to use some script languages like **MATLAB** or **OCTAVE** to test their ML algorithm first. After they see that it is working as expected they implement it with the language they actually need. They are in that way since MATLAB and OCTAVE are really good and well developed languages to implement such ML algorithm in really small number of lines. They includes lots of pre-developed math, linear and scientific functions that makes your job really easy. Thus I really suggest to use these two to test your ML program beforehand.



---

**Related posts:**

1. [Some Basic Machine Learning Terms #1](http://www.erogol.com/some-basic-machine-learning-terms-1/ "Some Basic Machine Learning Terms #1")