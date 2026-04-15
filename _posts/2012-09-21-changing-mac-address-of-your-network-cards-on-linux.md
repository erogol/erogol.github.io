---
layout: post
title: "Changing MAC address of your network cards on Linux"
description: "See the current MAC values

python
erogol@erogol-G50V ~ $ ifconfig -a | grep Ethernet
eth0      Link"
tags: computer security linux network
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

See the current MAC values

```python
erogol@erogol-G50V ~ $ ifconfig -a | grep Ethernet
eth0      Link encap:Ethernet  HWaddr 00:22:15:3a:36:93
wlan0     Link encap:Ethernet  HWaddr 00:15:af:dd:94:91
```

Stop running of the card

<pre ">**erogol@erogol-G50V ~ $**sudo ifconfig eth0 down

Change the MAC address

```python
erogol@erogol-G50V ~ $ sudo ifconfig hw ether 00:22:15:3a:36:83
```

Start the card running again

```python
erogol@erogol-G50V ~ $ sudo ifconfig eth0 up
```

That's all. You might see the new MAC address for check.

**Why I need to change the MAC address.**

* Protect your privacy on your using network. MAC layer is the most deepest layer that you can change any configuration. In that way if you change any setting, it makes you anonymous for all the other upper layers too. It is more effective than IP change (of course depends to the problem).
* It is used to acquire the connection on some local networks like internet connection in Starbucks. The following is the scenario. You need to investigate the connections in the cafe and see the MAC addresses on the header of packets going around. Set your MAC address to one of the recognized. In that way the local network in cafe will see you as another user already connected to the network because you are using his ID card, abstractly.
* To use torrent in torrent restricted networks like university campuses.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Simple hack to connect an authentication needed internet. (like in Starbucks)](http://www.erogol.com/simple-hack-to-connect-an-authentication-needed-internet-like-in-starbucks/ "Simple hack to connect an authentication needed internet. (like in Starbucks)")
2. [Hacker's first target file /etc/passwrd on Linux ! Why?](http://www.erogol.com/hackers-first-target-file-etcpasswrd-on-linux-why/ "Hacker's first target file /etc/passwrd on Linux ! Why?")