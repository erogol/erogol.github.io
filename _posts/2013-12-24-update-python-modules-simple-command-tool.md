---
layout: post
title: "Update all python modules with simple command tool"
description: "In case you use many modules all together, it is hard to keep track of latest versions and the requi"
tags: modules python tricks
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

In case you use many modules all together, it is hard to keep track of latest versions and the requisite updates. Therefore using such a little command regular might be useful.

```python
pip install pip-tools
$ pip-review --interactive
```

After some time, you observe that all the packages are updating.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [de-importing a Python module with a simple function](http://www.erogol.com/de-importing-python-module-simple-function/ "de-importing a Python module with a simple function")
2. [Two way python dictionary](http://www.erogol.com/two-way-python-dictionary/ "Two way python dictionary")
3. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")