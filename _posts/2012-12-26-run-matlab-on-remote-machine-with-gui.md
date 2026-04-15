---
layout: post
title: "Run Matlab On Remote Machine with GUI"
description: " Running Matlab Remotely: ssh -X

I wanted to run Matlab by logging into the university account remo"
tags: matlab remote machine ssh
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

### Running Matlab Remotely: ssh -X

I wanted to run Matlab by logging into the university account remotely from my machine. Everything went fine except for the graphics. Matlab started in no graphics mode.

After bit of a search I found the solution. You have to set the X11 forwarding in you ssh configuration file. Here's how to do it.

1. cd /etc/ssh  
2. sudo vi ssh\_config  
3. uncomment the lines "ForwardAgent" and "ForwardX11". Set their values to "yes"  
4. sudo vi sshd\_config  
5. uncomment "X11Forwarding" and set it value to "yes" as well.

that's it and you are good to go.

Type ssh -X username@domain

To test if everything works fine try running xclock once you log in. It should open up a graphical clock window.
