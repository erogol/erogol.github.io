---
layout: post
title: "Hacker's first target file /etc/passwrd on Linux ! Why?"
description: "At Linux /etc/passwrd file includes information about the user accounts on the operating system"
tags: computer security hack linux
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

At Linux /etc/passwrd file includes information about the user accounts on the operating system.  Permissions and password (if not encrypted) related with specific user account are stored here with some extra information. Here is the general structure of the file with the needed explanation to interpret it:

[![](http://files.cyberciti.biz/ssb.images/uploaded_images/passwd-file-783957.png)](http://files.cyberciti.biz/ssb.images/uploaded_images/passwd-file-791527.png)

1. **Username**: It is used when user logs in. It should be between 1 and 32 characters in length.
2. **Password**: An x character indicates that encrypted password is stored in /etc/shadow file.
3. **User ID (UID)**: Each user must be assigned a user ID (UID). UID 0 (zero) is reserved for root and UIDs 1-99 are reserved for other predefined accounts. Further UID 100-999 are reserved by system for administrative and system accounts/groups.
4. **Group ID (GID)**: The primary group ID (stored in /etc/group file)
5. **User ID Info**: The comment field. It allow you to add extra information about the users such as user's full name, phone number etc. This field use by finger command.
6. **Home directory**: The absolute path to the directory the user will be in when they log in. If this directory does not exists then users directory becomes /
7. **Command/shell**: The absolute path of a command or shell (/bin/bash). Typically, this is a shell. Please note that it does not have to be a shell.

Because passwrd file includes account information, a hacker that is able to inject in your system generally target the file (if he does not have any other special mission on). In that way he can get important account information about the system and use these to get the control on the system. However if your account passwords are kept as encrypted than they are stored in /etc/shadow file and to see that file you need to access the root password thus it is really hard (not impossible I guess) to get into that file for a average hacker.

/etc/shadow file has nearly same structure as /etc/passwrd but it includes the encrypted version of the passwords. In addition there is no standard encryption way but you can get some clue (as I search) from the first 3 char. of the encrypted password. For example if it starts with $1$, it means MD5-based algorithm is used.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Changing MAC address of your network cards on Linux](http://www.erogol.com/changing-mac-address-of-your-network-cards-on-linux/ "Changing MAC address of your network cards on Linux")
2. [Simple hack to connect an authentication needed internet. (like in Starbucks)](http://www.erogol.com/simple-hack-to-connect-an-authentication-needed-internet-like-in-starbucks/ "Simple hack to connect an authentication needed internet. (like in Starbucks)")