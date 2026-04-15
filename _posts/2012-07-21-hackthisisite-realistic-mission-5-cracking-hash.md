---
layout: post
title: "Hackthisisite realistic mission 5 - cracking hash"
description: "On that mission(http://www"
tags: cracking hacking hackthissite hashcat
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

On that [mission](http://www.hackthissite.org/playlevel/5/) you have a web site that has admin access to a email list and you want to acquire that access. On the explanation of the mission there are some key words.

> ...they used was 10 years out of date and the new password seems to be a '**message digest**'... I think it could be something like a so-called **hash value**. I think you could somehow reverse engineer it or **brute force** it...

So we need to find some hash value from somewhere on the site and use brute-force technique. (At this point you might want to understand hashing and message digest better from [wikipedia](https://en.wikipedia.org/wiki/Cryptographic_hash_function)). You might think for reversing hash value to password but is it really infeasible and it is the security of hashing. However you can brute-force if the password is not too long.

To start to solve the problem, you'll need a hash cracker something like [hashcat](http://hashcat.net/oclhashcat-plus/) that I used. (I used v.0.38)

* Go to target site's home page.
* Go to "news" page.
* There is a line saying that Google was crawling some unwanted files of the site and the owner resolved it. It measn there is a robot.txt file to direct google's crawler.
* Type "...realistic/5/robots.txt" to url robots.txt points two folders on server (/lib and /secret).
* Type "...realistic/5/lib" to url and you see the hash file. Download it.
* Open the hash file with a text editor. It shows lots of silly characters but if you look it carefully you might see the hashing algorithm that has been used. There are some lines that says MD4 algo. has been used for hashing the password. We'll use that information while cracking the hash.
* Then go to /secret folder and see the admin.bak.php file. If you click on it you will see the hash value that is expected to match with the password.
* Copy that value to a file and name it hash.txt. Store that file inside the hashcat folder.
* Than type that command to use hashcat inside the folder of the program.

`~. /hashcat-cli64.bin --hash-mode=900 --attack-mode=3 --bf-pw-min=1 --bf-pw-max=6  --bf-cs-buf=qwertyuiopasdfghjklzxcvbnmQWERTYUIOPLKJHGFDSAZXCVBNM1234567890 hash.txt`

To see the definitions of the options and the commands you might type ./hashcat-cli64.bin --help.

Also you would need to use ./hashcar-cli32.bin if you have 32 bit machine.  
After a while, program indicates the password  as the underlined part of below output.

f05f2d78b9ad41619649da7625bfe9a0:**2ab50**

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Sql injection - hack this site "Realistic Mission 2"](http://www.erogol.com/sql-injection-hack-this-site-realistic-mission-2/ "Sql injection - hack this site \"Realistic Mission 2\"")
2. [Directory Traversal Attack on Realistic Mission 3 HackThisSite.org](http://www.erogol.com/directory-traversal-attack-on-realistic-mission-3-hackthissite-org/ "Directory Traversal Attack on Realistic Mission 3 HackThisSite.org")
3. [.htaccess - basic 11 mission on hackthissite.com-](http://www.erogol.com/htaccess-basic-11-mission-on-hackthissite-com/ ".htaccess - basic 11 mission on hackthissite.com-")
4. [Good Way of having secure password...](http://www.erogol.com/good-way-of-having-secure-password/ "Good Way of having secure password...")