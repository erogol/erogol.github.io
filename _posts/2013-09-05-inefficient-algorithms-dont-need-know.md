---
layout: post
title: "Some inefficient algorithms you don't need to know!"
description: "Here we have some stupidly clever algorithms that are faster in execution but slower in convergence "
tags: algorithm c programming coding
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Here we have some stupidly clever algorithms that are faster in execution but slower in convergence to solution.

First candidate to introduce you is Bogo Sort. Its idea is simpler. Shuffle the numbers until it finds the correct order. Complexity of the algorithm can be approximated as O(n.n!) (efficient hah ?). Its pseudo code can be written as;

```python
while not isInOrder(deck):
    shuffle(deck);
```

Next one is known as Stooge Sort. The idea is to sort the initial 2/3 part then sort the final 2/3 part and cycle up to order of the numbers. Here is a 1/3 overlapping part as the guarantee the final correct order. However it is also very efficient sorting as the upper peer. Its complexity is O(n^2.7). The algorithm is written as;

```python
algorithm stoogesort(array L, i = 0, j = length(L)-1)
     if L[j] < L[i] then
         L[i] ↔ L[j]
     if (j - i + 1) >= 3 then
         t = (j - i + 1) / 3
         stoogesort(L, i  , j-t)
         stoogesort(L, i+t, j  )
         stoogesort(L, i  , j-t)
     return L
```

The last algorithm is might be far clever. I called it PrintSort. The main idea here is to print the item n after n secs later. In that way we will have printed array of numbers in sorted fashion. Flashy right! Its complexity is even linear O(n) since a single iteration is required to read all the numbers and print them all. It is a little hacky method that benefits from the famous O notation.

That's all for now. If you know any other dummy sorting method please let me know.

[Share](https://www.addtoany.com/share)

### Related posts:

1. [Sorting strings and Overriding std::sort comparison](http://www.erogol.com/sorting-strings-overriding-stdsort-comparison/ "Sorting strings and Overriding std::sort comparison")
2. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
3. [Setting up .sort function to your custom class in RUBY](http://www.erogol.com/setting-up-sort-function-to-your-custom-class-in-ruby/ "Setting up .sort function to your custom class in RUBY")
4. [Extracting a sub-vector at C++](http://www.erogol.com/extracting-sub-vector-c/ "Extracting a sub-vector at C++")