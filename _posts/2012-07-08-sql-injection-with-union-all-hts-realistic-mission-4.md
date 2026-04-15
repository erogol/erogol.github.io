---
layout: post
title: "SQL injection with UNION ALL : HTS realistic mission 4"
description: "> Fischer’s Animal Products: A company slaughtering animals and turning their skin into overpriced p"
tags: hackthissite
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

> **Fischer’s Animal Products**: A company slaughtering animals and turning their skin into overpriced products sold to rich bastards! Help animal rights activists increase political awareness by hacking their mailing list.

So I finally got around to write a walkthrough/guide for Hack This Site realistic mission 4. Your objective is to get the email addresses of the subscribers to the news letter of Fischer’s Animal Products.

> From: SaveTheWhales
>
> Message: Hello, I was referred to you by a friend who says you know how to hack into computers and web sites - well I was wondering if you could help me out here. There’s this local store who is killing hundreds of animals a day exclusively for the purpose of selling jackets and purses etc out of their skin! I have been to their website and they have an email list for their customers. I was wondering if you could somehow hack in and send me every email address on that list? I want to send them a message letting them know of the murder they are wearing. Just reply to this message with a list of the email addresses. Please? Their website is at http://www.hackthissite.org/missions/realistic/4/. Thanks so much!!

Start by investigating every part of Fischer’s the site. There are essentially two parts which might be vulnerable. The most visible one is the email form. A clearly visible input-field, where you just add your email address and are given a “Email added successfully” message. As you’ve seen through other missions containing [SQL injections](http://timjoh.com/hts-realistic-2-mysql-inject-the-nazi-party/), the first step is attempting to get out of the string. Try registering an email address containing apostrophes, both single and double.

> Error inserting into table “email”! Email not valid! Please contact an administrator of Fischer’s.

Unsuccessful. However, we got an important piece of information: the table name is “email”.

Now for the other part of the website; the product lists. There are two product lists, “fur coats” and “alligator accessories” (how this would have anything with whales to do is beyond me). If you’ve been as observant as you should be, you’ve noticed that both are the same file–products.php–with the category ID as an argument.

What do we want to accomplish? If we wanted to select something else from that table, we could attempt to change the WHERE part of the SELECT statement by changing the category argument to something like “1 OR categpory = 2″ (which happens to give you both categories of products on one page). However, we want to add information from another table: the “email” table. This is were the MySQL command [UNION](http://dev.mysql.com/doc/refman/5.0/en/union.html) comes in very handy. Using UNION, we can merge the results of two SELECT statements into one. For example, we could:

```python
SELECT * FROM table1 UNION ALL SELECT * FROM table2;
```

The result would be getting all rows from table1 and all rows from table2. Note that this assumes that the number of columns in table1 and table2 are equal. If they are not, the command will not work. UNION ALL is used instead of simply UNION in order to preserve duplicate rows. It is good practice to use UNION ALL in order to avoid unexpected errors. Let’s assume that the initial query could be something like this:

```python
SELECT * FROM products WHERE category = 1;
```

We also want the rows from the email table. Therefore, we’ll try looking for another category: `1 UNION ALL SELECT * FROM email`, resulting in the following final query:

```python
SELECT * FROM products WHERE category = 1 UNION
ALL SELECT * FROM email;
```

Which is exactly what we want. However, this results in nothing of value. Remember the assumption made earlier when we UNIONed table1 and table2? They must be of the same number of columns. We can assume that “email” has fewer columns than “products” does, since the products table should be more advanced. Therefore, we add columns to the email table:

```python
SELECT * FROM products WHERE category = 1 UNION
ALL SELECT *, NULL FROM email;
```

NULL means nothing–it is just an empty column. This doesn’t work either, so we’ll have to keep adding NULLs until we get some results. It will finally work at three NULLs:

```python
SELECT * FROM products WHERE category = 1 UNION
ALL SELECT *, NULL, NULL, NULL FROM email;
```

Below the category 1 products, you can see ten broken images. Viewing the source-code, you will find that the sources of these are email addresses! Rearranging the column order will give you a more eligible format.

```python
SELECT * FROM products WHERE category = 1 UNION
ALL SELECT NULL, *, NULL, NULL FROM email;
```

Just copy the list and email it to SaveTheWhales!

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [PHP parameter trick on hackthissite.org "Realistic Mission 1"](http://www.erogol.com/php-parameter-trick-on-hackthissite-org-realistic-mission-1/ "PHP parameter trick on hackthissite.org  \"Realistic Mission 1\"")
2. [What is SSI (server Side Includes)? -HackThisSite Basic 8 Solution.-](http://www.erogol.com/what-is-ssi-server-side-includes-hackthissite-basic-8-solution/ "What is SSI (server Side Includes)? -HackThisSite Basic 8 Solution.-")
3. [Cookie Hacking - hactkhissite basic10 -](http://www.erogol.com/cookie-hacking-hactkhissite-basic10/ "Cookie Hacking - hactkhissite basic10 -")
4. [Sql injection - hack this site "Realistic Mission 2"](http://www.erogol.com/sql-injection-hack-this-site-realistic-mission-2/ "Sql injection - hack this site \"Realistic Mission 2\"")