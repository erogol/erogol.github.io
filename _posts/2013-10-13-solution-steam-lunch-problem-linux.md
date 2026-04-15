---
layout: post
title: "Solution to Steam lunch problem in linux"
description: "If you installed Steam to your linux machine and see some lunch alerts complaining about such error;"
tags: linux problems solution ubuntu
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

If you installed Steam to your linux machine and see some lunch alerts complaining about such error;

> You are missing the following 32-bit libraries, and Steam may not run: libGL.so.1

this is your solution time. Here is the solution.

This seem to happen on every 64bits OS. Full bug report here:

<https://github.com/ValveSoftware/steam-for-linux/issues/321>

Solution:

sudo gedit /etc/ld.so.conf.d/steam.conf

Add next two lines to file:

/usr/lib32  
/usr/lib/i386-linux-gnu/mesa

Then execute:

sudo ldconfig

Now you should be able to run Steam

---

**Related posts:**

1. [Creating Custom Linux Command](http://www.erogol.com/creating-custom-linux-command/ "Creating Custom Linux Command")
2. [CUDA compilation error and the solution](http://www.erogol.com/cuda-compilation-error-and-the-solution/ "CUDA compilation error and the solution")
3. [Today’s Linux Console Command](http://www.erogol.com/todays-linux-console-command/ "Today’s Linux Console Command")
4. [Some console commands on linux.](http://www.erogol.com/some-console-commands-on-linux/ "Some console commands on linux.")
5. [Seeing Your Local DNS Server IP Address in Linux](http://www.erogol.com/seeing-your-local-dns-server-ip-address-in-linux/ "Seeing Your Local DNS Server IP Address in Linux")