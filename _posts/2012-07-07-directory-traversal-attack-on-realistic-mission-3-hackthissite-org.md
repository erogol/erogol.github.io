---
layout: post
title: "Directory Traversal Attack on Realistic Mission 3 HackThisSite.org"
description: "What I learn from HTS today is Directory Traversal Attack (DTA)"
tags: hacking hackthissite
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

What I learn from HTS today is Directory Traversal Attack (DTA).  You might learn from [Wikipedia](https://en.wikipedia.org/wiki/Directory_traversal_attack). As a summary DTA is a way of accessing the locations that are not intended to be available to plain user, by using input fields of the website. Generally flaw that makes open to DTA is low sanitizing and input validation of applications.

These are the steps to complete the mission.

1. Open the hacked index of the web site and open the source of the index see the bottom comment of the hackers. It means we have original index file as oldindex.html
2. Type to ...3/oldindex.html
3. Open the source and copy all the source of the page.
4. Go to "Submit Poetry" page of the site.
5. Type ../index.html as name and paste all the copied content to content part of the form.
6. submit. That's all 🙂

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Sql injection - hack this site "Realistic Mission 2"](http://www.erogol.com/sql-injection-hack-this-site-realistic-mission-2/ "Sql injection - hack this site \"Realistic Mission 2\"")
2. [Hackthisisite realistic mission 5 - cracking hash](http://www.erogol.com/hackthisisite-realistic-mission-5-cracking-hash/ "Hackthisisite realistic mission 5 - cracking hash")
3. [Cookie Hacking - hactkhissite basic10 -](http://www.erogol.com/cookie-hacking-hactkhissite-basic10/ "Cookie Hacking - hactkhissite basic10 -")
4. [Good Way of having secure password...](http://www.erogol.com/good-way-of-having-secure-password/ "Good Way of having secure password...")