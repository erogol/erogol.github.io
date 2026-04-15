---
layout: post
title: "PHP MVC Frameworks"
description: "Zend Framework is a good foundation for everything and can be used also as just a library of various"
tags: coding famework mvc php programming
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Zend Framework** is a good foundation for everything and can be used also as just a library of various functions.  It is also the closest thing to be  an "official" PHP framework so there are a lot of developers who know how to use it.  It is, however, not a framework you would use to prototype something very quickly.

There are several frameworks that could be used as a quick RAD (Rapid Application Development) tools.

A good example is **CakePHP**, which is a very popular framework.  Fairly easy to learn.  It has a lot of sensible defaults and naming conventions that make your life much easier, but which you can override.

I was pretty skeptical about the viability of the CakePHP project since some of the core developers have left in 2009 to form the Lithium Framework.  However it got particular boost with the development release of CakePHP 2.0 which will address a number of issues developers had with the original CakePHP.  Overall, this should be an interesting framework to watch.

Another one in this category is **CodeIgniter** and it's another reincarnation called **Kohana** (which was initially started as a CodeIgniter port to PHP 5, but grew up significantly since than) -- these are pretty solid development frameworks and quite popular ones, so I would recommend to take a look at one of those.

There is also increasingly popular **Yii** framework, which is also good for development of dynamic, AJAX heavy applications.

Than there are bigger frameworks designed for more advanced professionals offering a lot of functionality but with a steeper learning curve.

Good example in this category is **Symfony** -- very popular and generally well build one.  They are relaunching it as Symfony2 -- completely rewritten for PHP 5.3.  Symfony also promotes heavily usage of the Doctrine ORM for data access (actually they also release Doctrine2 for PHP 5.3).  Alternatively it also allows you to use another ORM called Propel.

Another emerging framework in this category is called **Lithium Framework** -- also written exclusively for PHP 5.3.  This one is made by the original developers of CakePHP, who decided to start the newer more advanced framework from the scratch.  Lithium uses heavily the new features in PHP 5.3, including namespaces, closures and late static binding and look very interesting from the design perspective.  I definitely recommend to at least look at this one, even if to learn a thing or two about good coding style.  It also has it's own database abstraction layer similar to ActiveRecord.

All of the frameworks listed have MVC, templating and database abstraction layer (either internal or use the widely accepted ORM systems like Doctrine or Propel.)

Most of them based on the standard PDO (PHP Database Object) library that comes with PHP.  BTW, even if you do not use any of the frameworks, it is generally a good practice to use PDO for database access as it is improves the readability and portability of your code with fairly modest overhead.

Here are few links for you to check:  
 <http://cakephp.org/>  
 <http://codeigniter.com/>  
 <http://kohanaframework.org/>  
 <http://www.yiiframework.com/>

<http://www.symfony-project.org/> -- Symfony  
 <http://symfony-reloaded.org/> -- Symfony2  
 <http://lithify.me/> -- Lithium Framework

<http://www.propelorm.org/> -- Propel ORM  
 <http://www.doctrine-project.org/> -- Doctrine ORM

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [My comments about PHP+Smyfony2, RoR, PHP+WordPress](http://www.erogol.com/my-comments-about-phpsmyfony2-ror-phpwordpress/ "My comments about PHP+Smyfony2, RoR, PHP+WordPress")