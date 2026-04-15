---
layout: post
title: "Intallation CUDA to Ubuntu 12.10 with Optimus Nvidia Cards"
description: "I installed Ubuntu 12"
tags: cuda installation linux nvidia
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I installed Ubuntu 12.10 to my brand new machine but as always I started to deal lot of deriver problems coming around. Most consuming trouble was about the Nvidia drivers. I installed all kind of drivers suggested by the Additional Drivers tool and the Nvidia website but I cannot get my Graphic card working. After hours of investigation see that with the new generation notebooks with Nvidia cards there is a new technology called Optimus. With that system, new machines have two different graphic cards as the Intel’s native one on the mother board and Nvidia Card. To prolong the battery life, Intel card is working for simple graphic rendering where as Nvidia comes into play with hard rendering problems so that machine can keep the battery life better in hours. However, Nvidia is deficient to provide a driver supporting new tech on Linux machines. As always solution is taken by the open source approach, [Bumblebee](http://bumblebee-project.org/index.html) driver interface is developed. In order to make your card working at Ubuntu you need to install bumblebee drivers and use ‘optirun appname’ command to utilize Nvidia card. If you don not run your app with optirun, Intel card will handle.  To get moe detail abut Bumblebee follow the above link.

Now what is the problem about CUDA. Actually installing Bumblebee is not directly the solution for the problem. Since we use Ubuntu 12.10 and CUDA currently support officially only the 11.10 version, we need to do some minor changes. For my installation, changing the native gcc compilers from 4.7 version to 4.4 version has solved the problem. 12.10 comes with 4.7 version of compilers but CUDA installation needs 4.4 version.

I find an bash script and have some little changes over (since it was not enough to handle all compilation at my case). Here is the [script at github.](https://github.com/erogol/myBashScripts) This script will handle all the needs of intallation. All you have to do that downloading CUDA installation file from [NVIDIA](http://www.nvidia.com/Download/index.aspx?lang=en-us) (if you are on 12.20 64 bit than install 11.10 64bit version) and retyping the  installation file path on the script. Than run the script with ‘bash’ command. Thats all!

**Caveats:**

Do not install suggested driver from the CUDA installation file. It destroys the Bumblebee configuration on system.

If you have any trouble or question let me know.

After installation you need to run CUDA executables with ‘**optirun** ./executable.out’

---

**Related posts:**

1. [Fixind the DropBox after updating Ubuntu to version 11.04 with Unity.](http://www.erogol.com/fixind-the-dropbox-after-updating-ubuntu-to-version-11-04-with-unity/ "Fixind the DropBox after updating Ubuntu to version 11.04 with Unity.")
2. [Changing MAC address of your network cards on Linux](http://www.erogol.com/changing-mac-address-of-your-network-cards-on-linux/ "Changing MAC address of your network cards on Linux")
3. [Simple hack to connect an authentication needed internet. (like in Starbucks)](http://www.erogol.com/simple-hack-to-connect-an-authentication-needed-internet-like-in-starbucks/ "Simple hack to connect an authentication needed internet. (like in Starbucks)")
4. [Installing MySQL Server to Ubuntu](http://www.erogol.com/installing-mysql-server-to-ubuntu/ "Installing MySQL Server to Ubuntu")
5. [Installing MySql to Ubuntu](http://www.erogol.com/installing-mysql-to-ubuntu/ "Installing MySql to Ubuntu")