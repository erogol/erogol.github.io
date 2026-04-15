---
layout: post
title: "ML Work-Flow (Part 4) – Sanity Checks and Data Splitting"
description: " SANITY CHECK

We are now one step ahead of Feature Extraction(http://www"
tags: data_splitting machine learning ml path-way sanity_check
minute: 6
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

# SANITY CHECK

We are now one step ahead of [Feature Extraction](http://www.erogol.com/ml-work-flow-part-3-feature-extraction/) and we extracted statistically important (covariate) representation of the given raw data. Just after Feature Extraction, first thing we need to do is to check the values of the new representation. In general, people are keen on avoiding this and regarding it as a waste of time. However, I believe this is a serious mistake. As I stated before, a single  NULL value, or skewed representation might cause a very big pain at the end and it can leave you in very hazy conditions.

Let’s start our discussion. I list here my Sanity Check steps;

![](http://lowres.cartoonstock.com/computers-date_analyst-data_analysis-expectations-flaws-data_entry-aban1402_low.jpg)

**Check for NULL values an understand why they are NULL** - NULL values are informative even they blight you ML pipeline. They are indicators of  the problems sparked by  the preceding stages. Therefore before dwelling into the problem more techically, these NULL values enable you to solve these problems in advance.

If you observe NULL values just after Feature Extraction there are some common issues to consider;![](http://theaccessbuddy.files.wordpress.com/2012/10/image8.png)

* **Fed data is not in the expected format** of the Feature Extraction algorithm. Incidentally, particular Feature Extraction methods demand certain Normalization, Standardization, Scaling procedures over the raw data or maybe you might need to change the value types via Discretization, Categorization or such. I'll talk about these particular procedures at Feature Preprocessing stage but they have some use cases before, as suggested here. If you are fine with the data format then consider to use a different code to extract feature and compare the results with the first one. If something is wrong and they are not concordant then use a third code to rectify the correctness. One caveat is whether the algorithm is one of the deterministic approaches. If it is not,for each run you might observe different values and it is reasonable. In that case, it is better to visualize the features in some way and try to see the expected semantics. For instance, for Feature Learning approaches in the context of Computer Vision, display learned filters and justify visual correctness. For other domains, research through the community and find a way to do so.

* **Your ETL work-flow is rotten**. ETL is a process of merging data from different resources through some software or simple code flows. This process should be very adaptable to internal changes of the data resources. For example, if you merge data from two different DBs, then structural change on a table is able to blow your ETL process and that causes new NULL values. This is a very frequent experience for me as well. Therefore, ETL process needs to be robust to these changes or at least it should log such problems.![](http://static.fjcdn.com/pictures/Division_62e0fe_1196509.jpg)
* **Zero division 🙂**

**Check the value scale.** MAX and MIN values, BOX plots, Scatter plots, Mean-Median differences are useful pointers to skewed values. Plot these and observe whether they are reasonable or not. If there is something that seems wrong, investigate it. Investigation is likely to demand some level of expertise.

**Check number of unique values for each dimension.** This sounds very stupid but believe me, this is a very useful method for checking the values. Even when your data is categorical, nominal or continuous, this is always very useful anyway. Plot a bar chart that depicts the number of different values for each data dimension.

# **DATA SPLITTING**

Okay, we rectified the correctness of our data representation after Feature Extraction. Now, it is time to split our data into test, train and optionally held-out and validation sets.

* **Train-Set** : Train your models with this.
* **Validation-Set**: Particular algorithms like Neural Networks are better to use separate Validation-Set to see generalization performance in the training cycles. Because  these algorithms are very likely to over-fit,using training data for the only measure of the training time can be misleading to extreme over-fitting. Instead, use the Validation-Set for each iteration performance and stop training where training and validation values disperse at some level. Another use-case of Validation-Set is to use for performance measurement where cross-validation is not scalable. If you are toying with BigData with a low budget system then a simple cross-validation takes days or weeks. Therefore, for simple experiments like simple settings changes or first insights of the proposed algorithms, use Validation-Set. After you get these values and make initial decisions, you force yourself to apply cross-validation with the mixture of train and validation data. That is my recommended setting.
* **Test-Set** : This should be used after you are done with your model after Train-Set and Validation-Set. Test-Set measures the performance after you set all the parameters of your model according to train and validation steps. Since you cycle around train and validation repeatedly, your model might also over-fit not only train data but validation data as well. From this respect, Test-Set shows the level of **rational over-fitting** of you. This is why most of the Kaggle competitions measures only the final performances with the whole data that mimics our Test-Set notion.
* **Held-Out Set:** This might be combination of the other sets. However, the best of all is to use purely separate data, unless you have data shortage. Cardinally, this data is not touched before you say “I am done! This is my masterpiece”. Hence, you measure the final score of your model at this set and this is the best performance approximation of the real world scenario. Since you have not touched it before and you have not reiterated your model by through this set, your model has no sight of bias about Held-Out Set. Therefore, it is your final performance measurement that you report.

I believe many people will be reluctant for my proposed data splitting scheme and they will consider this level of granularity meaningless against simple train, test split + cross-validation scheme. However, I assure them, especially in the industry this formation is the best against the idea of rational over-fitting. By the lights of these argument, I also believe that present notion of data-splitting is an important crux of the academy. We are exposed to so called state-of-art methods telling very high accuracies with very data specific, rationally over-fitted methodologies. As a matter of that fact, when we apply these methods to real time problems, values reduce too much to be true. Yes, models can over-fit the given train data but using only a limited data-set with certain train and validation cycle also inclines you to over-fit the given problem rationally. One example of rational over-fitting is the Neural Network structures used for [Image-Net](http://image-net.org/index) competitions. If you look to the papers or each year's methods, you will see that people still use the same topology of [Alex's Net](https://code.google.com/p/cuda-convnet/) (first important Deep Learning success in Image-Net) over years and declares 0.5% improvement with little or less methodological improvements. Incidentally, we have a huge literature of Deep Learning community over-fitted to Image-Net. My solution to this is to use separate data-set for ranking each year's competition by training models from the last year dataset. In my opinion, this would be a more robust adjustment against this kind of over-fitting of course with more annotation complexity.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [ML Work-Flow (Part 3) - Feature Extraction](http://www.erogol.com/ml-work-flow-part-3-feature-extraction/ "ML Work-Flow (Part 3) - Feature Extraction")