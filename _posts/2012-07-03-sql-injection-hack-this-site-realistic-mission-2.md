---
layout: post
title: "Sql injection - hack this site \"Realistic Mission 2\""
description: "Today it is the turn for the realistic mission 2 on hackthissite"
tags: hacking hackthissite realistic mission sql injection
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Today it is the turn for the realistic mission 2 on [hackthissite.org](http://www.hackthissite.org/).

This mission is all about looking the home page source code. Finding the hidden link on page to directs you to admin page then use basic SQL injection to accomplish the mission.

SQL injection is about typing some malformed values to html forms to make some changes on the application database or get some data that the application owner does not expect us to see them or change. You can learn more about SQL injection from [this link.](http://unixwiz.net/techtips/sql-injection.html)

You need to be able to pass the mission after all the explanation and the reading from the above reference site. If you cannot, it means you need to work some more on hacking the sites. However for the lazy brains here I give the instructions as follows:

1. Open the source file of the page.
2. See the update.php link on the source. It is hidden on the visuals on the page.
3. Find the hidden link and click on it to go to admin login page.
4. Now use one of the tricks that you know about sql injection. I used this for both input  x' OR 1 = 1;

That's all ![:)](http://www.erogol.com/wp-includes/images/smilies/simple-smile.png)



---

**Related posts:**

1. [PHP parameter trick on hackthissite.org "Realistic Mission 1"](http://www.erogol.com/php-parameter-trick-on-hackthissite-org-realistic-mission-1/ "PHP parameter trick on hackthissite.org  \"Realistic Mission 1\"")