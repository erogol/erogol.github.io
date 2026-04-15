---
layout: post
title: "Setting up cudamat in Ubuntu Machine"
description: "cudamat( http://code"
tags: cuda machine learning python
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

[cudamat]( http://code.google.com/p/cudamat/) is a python library that makes you available to use CUDA benefits from Python instead of intricate low level approaches. This interface uses also

Before follow these steps please make sure that you installed a working CUDA library.

1. Download [cudamat]( http://code.google.com/p/cudamat/) from
2. Compile with 'make' in the root downloaded folder /path/to/cudamat
3. Set the environment variables to include cudamat in PYTHONPATH to be able to imported by any script. Run followings in the command line.

   ```python
   
    PYTHONPATH=$PYTHONPATH:/path/to/cudamat
    export PYTHONPATH
   ```
4. You are ready to use cudamat.

Here is a simple code you might test;

```python

 # -*- coding: utf-8 -*-
 import numpy as np
 import cudamat as cm
 cm.cublas_init()
 # create two random matrices and copy them to the GPU
 a = cm.CUDAMatrix(np.random.rand(32, 256))
 b = cm.CUDAMatrix(np.random.rand(256, 32))
 # perform calculations on the GPU
 c = cm.dot(a, b)
 d = c.sum(axis = 0)
 # copy d back to the host (CPU) and print
 print d.asarray()

```

**Note**: If you get any other path problem, it would be related to CUDA installation therefore check environment parameters need to be set for CUDA.



---

**Related posts:**

1. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
2. [Anomaly detection and a simple algorithm with probabilistic approach.](http://www.erogol.com/anomaly-detection-and-a-simple-algorithm-with-probabilistic-approach/ "Anomaly detection and a simple algorithm with probabilistic approach.")
3. [Data-Driven Enhancement of Facial Attractiveness](http://www.erogol.com/data-driven-enhancement-of-facial-attractiveness/ "Data-Driven Enhancement of Facial Attractiveness")
4. [Some possible ways to faster Neural Network Backpropagation Learning #1](http://www.erogol.com/some-possible-ways-to-faster-neural-network-backpropagation-learning-1/ "Some possible ways to faster Neural Network Backpropagation Learning #1")
5. [What is special about rectifier neural units used in NN learning?](http://www.erogol.com/what-is-special-about-rectifier-neural-units-used-in-nn-learning/ "What is special about rectifier neural units used in NN learning?")