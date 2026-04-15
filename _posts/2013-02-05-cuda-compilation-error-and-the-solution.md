---
layout: post
title: "CUDA compilation error and the solution"
description: "python"
tags: cuda error solution
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

```python
./clock: error while loading shared libraries: libcudart.so.5.0: cannot open shared object file: No such file or directory
```

**Solution**

```python
sudo ldconfig /usr/local/cuda/lib64^C
```

---

**Related posts:**

1. [Gem Error while try to execute Rails commads](http://www.erogol.com/gem-error-while-try-to-execute-rails-commads/ "Gem Error while try to execute Rails commads")
2. [Intallation CUDA to Ubuntu 12.10 with Optimus Nvidia Cards](http://www.erogol.com/intallation-cuda-to-ubuntu-12-10-with-optimus-nvidia-cards/ "Intallation CUDA to Ubuntu 12.10 with Optimus Nvidia Cards")
3. [Run Matlab On Remote Machine with GUI](http://www.erogol.com/run-matlab-on-remote-machine-with-gui/ "Run Matlab On Remote Machine with GUI")
4. [Creating Custom Linux Command](http://www.erogol.com/creating-custom-linux-command/ "Creating Custom Linux Command")
5. [Compiling Kernel Method 2 (As Debian Package)](http://www.erogol.com/compiling-kernel-method-2-as-debian-package/ "Compiling Kernel Method 2 (As Debian Package)")