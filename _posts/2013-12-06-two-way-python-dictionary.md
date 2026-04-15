---
layout: post
title: "Two way python dictionary"
description: "Here there is a simple class implementation of Two way Dictionary that uses from native dictionary c"
tags: code python tip
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Here there is a simple class implementation of Two way Dictionary that uses from native dictionary class of Python. The main idea of that kind of data structure to reach, but side of the data by using other side as the key. It is like bi-directional relation between items.

A dumb use case:

```python
d = TwoWayDict()
d['erogol'] = 13
print d['erogol']
# outputs 13
print d[13]
# outputs erogol
```

Here is the class implementation. However, keep in mid that this class uses more x2 memory to keep the data with that functionality.

```python
class TwoWayDict(dict):
    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)
```

Hope this helps in some hinge 🙂

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [de-importing a Python module with a simple function](http://www.erogol.com/de-importing-python-module-simple-function/ "de-importing a Python module with a simple function")
2. [Update all python modules with simple command tool](http://www.erogol.com/update-python-modules-simple-command-tool/ "Update all python modules with simple command tool")
3. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
4. [Passing multiple arguments for Python multiprocessing.pool](http://www.erogol.com/passing-multiple-arguments-python-multiprocessing-pool/ "Passing multiple arguments for Python multiprocessing.pool")