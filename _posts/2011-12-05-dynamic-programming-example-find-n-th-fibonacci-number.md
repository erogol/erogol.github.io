---
layout: post
title: "Dynamic Programming Example - Find n.th Fibonacci number -"
description: "Dynamic programming is a concept that provides faster solutions for divide and conquer problems that"
tags: dynamic-programming fibonacci java
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Dynamic programming is a concept that provides faster solutions for divide and conquer problems that have some number of overlapping sub problems. Generally this concept is used for optimization problems like finding longest common sub-sequence of two arrays.

However I used the concept in finding n th Fibonacci number. Here is the code that I used.

```python
public class Main {
	int q;
	public static void main(String args[]){
		new Main().fibonacci(40);
	}

	public void fibonacci(int n){
		ArrayList r = new ArrayList();

		for(int i = 0; i < n+1; i++){
			r.add(-1);
		}
		fibonacci(n, r);
		System.out.println(q);
	}

	public int fibonacci(int n, ArrayList r){
		if (r.get(n)>= 0){
			return r.get(n);
		}else if(n == 0){
			q = 0;
		}else if(n == 1){
			q = 1;
		}else{
			q = fibonacci(n-1, r)+fibonacci(n-2, r);
		}
		r.set(n, q);
		return q;
	}
}
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [A solution for a Greedy Choice Algorithm Solution from TopCoder](http://www.erogol.com/a-solution-for-a-greedy-choice-algorithm-solution-from-topcoder/ "A solution for a Greedy Choice Algorithm Solution from TopCoder")
2. [My Multi Threaded Crawler (JAVA)](http://www.erogol.com/my-multi-threaded-crawler-java/ "My Multi Threaded Crawler (JAVA)")
3. [Project Euler - Problem 14](http://www.erogol.com/project-euler-problem-14/ "Project Euler - Problem 14")