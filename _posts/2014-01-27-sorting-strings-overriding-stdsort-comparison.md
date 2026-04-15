---
layout: post
title: "Sorting strings and Overriding std::sort comparison"
description: "At that post, I try to illustrate one of the use case of comparison overriding for std::sort on top "
tags: algorithm c c programming
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

At that post, I try to illustrate one of the use case of comparison overriding for std::sort on top of a simple problem. Our problem is as follows:

> Write a method to sort an array of strings so that all the anagrams are next to each other.

It seems very complicated at the first sight but if you know that little trick then it is very easy to grasp.

What we're gonna do is :

1. sort each string chars in itself before comparison.
2. compare each pair of strings.
3. sort with that comparison.

The realization of the algorithm is below with clearing comments.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [What is "long long" type in c++?](http://www.erogol.com/what-is-long-long-type-in-c/ "What is \"long long\" type in c++?")
2. [Some inefficient algorithms you don't need to know!](http://www.erogol.com/inefficient-algorithms-dont-need-know/ "Some inefficient algorithms you don't need to know!")
3. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
4. [Extracting a sub-vector at C++](http://www.erogol.com/extracting-sub-vector-c/ "Extracting a sub-vector at C++")