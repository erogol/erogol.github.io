---
layout: post
title: "Short Notes on GPT-3"
description: "Original paper :<https://arxiv"
tags: deep-learning gpt-3 language-model machine learning nlp
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Original paper :<https://arxiv.org/abs/2005.14165>

* It uses the same architecture as GPT-2.
* The largest model uses **170B parameters** and trained with a batch size of **3.2 million**. (Wow!).
* Training cost exceeds $12M.
* “Taking all these runs into account, the researchers estimated that building this model [generated](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/) over 78,000 pounds of CO2 emissions in total—**more than the average American adult will produce in two years**.”[[link](https://www.forbes.com/sites/robtoews/2020/06/17/deep-learnings-climate-change-problem/#c644ae46b438)]
* They used a system with more than **285K CPU cores** **10K GPUs** and 400 Gigabits network connectivity per machine. (Too much pollution).
* The model is trained on the whole Wikipedia, 2 different Book datasets, and Common Crawl.
* It learns different tasks with a task description, example(s), and a prompt.
  + **Task description** is a definition of target action, like “Translate from English to France…”
  + **Example(s)** is a sample or a set of samples used in one-shot or few-shot learning settings.
  + **Prompt** is an input on which the target action is performed.
* The larget the model, the better the results.
* They perform zero-shot, one-shot, and few-shot learning with the pre-trained language model for specific tasks.
* At **the Question Answering** task, it outperforms SOTA models trained with the source documents.
* At **the Translation** task, it performs close to SOTA. It is better at translating a language to English than otherwise, given it is trained on an English corpus.
* **Winograd task** is determining which word a pronoun refers to in a sentence.
* Physical Q&A is asking questions about grounded knowledge about the physical world. Outperforms SOTA in all the learning settings.
* Reading Comprehension is asking questions about a given document. Performs poorly in relation to SOTA.
* Causal Reasoning is giving a sentence and asking the most possible outcome.
* Natural Language Interference is a task to determine if the 2nd sentence is matching or conflicting with the 1st sentence. It performs well here.
* at **arithmetic operations,** small models perform poorly and large models perform good, especially at summation. They discuss that multiplication is a harder operation.
* At **SAT**, it performs better than an average student.
* Human accuracy to **detect the articles** written by the model is close to the random guess with the largest model.

I believe GPT-3 is not capable of “reasoning” in contrary to the common belief. GPT-3 rather constitutes an efficient storage mechanism for data it is trained with. At inference time, the model determines the output by finding the samples that are most relevant to the given task and interpolating them.

This is more apparent at the arithmetics task. The summation task is much easier since it is easier to memorize the whole table of information from the training data. And as it gets harder with multiplication the model struggles to fetch the relevant information and the performance drops.

You can also observe that when you take sentences from the generated articles in the paper and google them. Although they do not exactly match any article on the Web, you see very similar content and sometimes sentences that are different by only a couple of words.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.