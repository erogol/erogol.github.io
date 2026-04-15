---
layout: post
title: "What is Metaprogramming?"
description: "Metaprogramming is coding some programs that generates new code segments to be executed while execut"
tags: 
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Metaprogramming is coding some programs that generates new code segments to be executed while execution time. So why it is a need.

* We determined which problems were best solved with a code-generating program, including:
  + Programs that need to pre-generate data tables
  + Programs that have a lot of boilerplate code that cannot be abstracted into functions
  + Programs using techniques that are overly verbose in the language you are writing them in
* We then looked at several metaprogramming systems and examples of their use, including:
  + Generic textual-substitution systems
  + Domain-specific program and function generators
* We then examined a specific instance of table-building
* We then wrote a code-generating program to build static tables in C
* Finally, we introduced Scheme and saw how it is able to tackle the issues we faced in the C language using constructs that were part of the Scheme language itself

I got familiar to this idea from the talk of a ruby programmer and here is the [link](http://www.infoq.com/presentations/metaprogramming-ruby) to the video of that talk about how to metaprogramming in ruby.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.