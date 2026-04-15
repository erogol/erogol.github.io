---
layout: post
title: "Run matlab codes from terminal."
description: "Sometimes it is necessary to run your matlab codes from terminal, likely when you are away on remote"
tags: commands matlab terminal tricks
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Sometimes it is necessary to run your matlab codes from terminal, likely when you are away on remote connection to your work station. Sometimes you need to run couple of matlab instances from same terminal with additional & sign at the end of the terminal command. Now I'll show a basic command to be able to run youR \*.m script from terminal.

matlabOn the terminal you and the directory where the matlab bin file is located you type that command.

```python
./matlab -nodesktop -nosplash -r "run path/to/your/*.m/file"
```

DO NOT USE  .m FILE EXTENSION JUST LIKE YOU ARE CALLING A SCRIPT AT NATIVE MATLAB ENVIRONMENT.

After -r macro at the quote signs every thing you write is respected as a normal matlab code thus you might write any other execution sequence you want.



---

**Related posts:**

1. [Run Matlab On Remote Machine with GUI](http://www.erogol.com/run-matlab-on-remote-machine-with-gui/ "Run Matlab On Remote Machine with GUI")