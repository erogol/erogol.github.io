---
layout: post
title: "Using IPython and IPDB for Debugging"
description: "This is one of the things I always need but I forget"
tags: breakpoint debugging ipdb ipython python
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

This is one of the things I always need but I forget. So here is a piece of mind to check back.

```python
pip install ipython 
pip install ipdb
export PYTHONBREAKPOINT=ipdb.set_trace  # this is to use ipdb by default
```

When you run your code with a breakpoint, you get an IPython shell for debugging. So you can use all the perks like autocomplete, magic functions, etc.

```python
print("this is an example")
breakpoint()
print("this is the end")
```

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2021/11/Screenshot-2021-11-17-at-11.59.01.png)

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.