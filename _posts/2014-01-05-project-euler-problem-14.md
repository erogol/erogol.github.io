---
layout: post
title: "Project Euler - Problem 14"
description: "Here is one again a very intricate problem from Project Euler"
tags: algorithms dynamic-programming project-euler
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Here is one again a very intricate problem from Project Euler. It has no solution sheet as oppose to the other problems at the site. Therefore there is no consensus on the best solution.

Below is the problem: (I really suggest you to observe some of the example sequences. It has really interesting behaviours. 🙂 )

> The following iterative sequence is defined for the set of positive integers:
>
> n ![→](http://projecteuler.net/images/symbol_maps.gif) n/2 (n is even)  
>  n ![→](http://projecteuler.net/images/symbol_maps.gif) 3n + 1 (n is odd)
>
> Using the rule above and starting with 13, we generate the following sequence:
>
> 13 ![→](http://projecteuler.net/images/symbol_maps.gif) 40 ![→](http://projecteuler.net/images/symbol_maps.gif) 20 ![→](http://projecteuler.net/images/symbol_maps.gif) 10 ![→](http://projecteuler.net/images/symbol_maps.gif) 5 ![→](http://projecteuler.net/images/symbol_maps.gif) 16 ![→](http://projecteuler.net/images/symbol_maps.gif) 8 ![→](http://projecteuler.net/images/symbol_maps.gif) 4 ![→](http://projecteuler.net/images/symbol_maps.gif) 2 ![→](http://projecteuler.net/images/symbol_maps.gif) 1
>
> It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
>
> Which starting number, under one million, produces the longest chain?
>
> **NOTE:** Once the chain starts the terms are allowed to go above one million.

The difficulty of the problem lies over the jingling nature of the problem. What I mean by that, initially it seems like a Dynamic Programming problem,  with overlapping sub-problems. That is, in order to find the max length sequence, seeded by a number in the given range, crumble down the larger problem into smaller pieces . Find longest sequence for more scanty range as a sub-problem and incrementally search for optimal one on larger range. However, problem values are not very predictable since they are not monotonically decreasing. If we encounter a odd number, next number might jump out to the unpredictable values out of the 1 million range.  Therefore it is hard (at least) for me to accommodate that problem onto DP view.

Nevertheless, my method devices Memorization based approach. Memoization is a intermediate layer between greedy and DP. It does not solve all the sub-problems as DP or it does not take best choice of the present step. Instead you write out all the computation to a vocabulary and use those results at the forthcoming steps instead of recomputation ( For more <https://en.wikipedia.org/wiki/Memoization>  
 ). Notice that, also my solution is not exact Memoization since convenient Memoization approach requires Top-Down recursion.

Here is what I did in words, headed by the matching functions;

**memo\_seed\_length :** It finds the length of the sequence seeded by given value. It grasps 2 arguments as seed and the K is the vocabulary that we keep sequence lengths computer previously. The main idea of that recursive function is to compute the sequence length up to a know value in the K. If we find a computed K entry then return that value and add +1 for each recursion layer (for each sequenc entry).

**bottom\_up\_memoization :** After initialize the K list, it fills trivial values into. Those values are the entries that are corresponding to 2's exponents. Those values are known to be divided by 2 up to 1 without. For instance thinks about $16 = 2^4 $. It is divided by 2 for all along its sequence.  At the last part of the function, it just runs the upper function for each values from 1 to 1 million. After this final iteration, I end up a fully loaded list K and its largest value is the answer of the problem.

Here is the code below:

```python
# auxilliary functions
def check_even( num ):
	return num % 2 == 0

def check_odd( num ):
	return ( num + 1 ) % 2 == 0

def apply_odd( num ):
	return ( 3* num + 1)

def apply_even( num ):
	return ( num / 2 )

def get_next_num( num ):
	if check_even( num ):
		new_num = apply_even( num )
	else:
		new_num = apply_odd( num )
	#print new_num
	return new_num

def memo_seed_length(seed, K, seq = None):
	if seed == 1:
		K[0] = 1
		return 1
	elif seed < len(K) and K[seed-1] != -1:
		return K[seed-1]
	else:
		next_seed =  get_next_num(seed)
		seq_len = 1 + memo_seed_length(next_seed, K)
		if seed < len(K):
			K[seed-1] = seq_len
		return seq_len

def bottom_up_memoization(upper_lim):
	K = [-1]*upper_lim

	expo = 0
	while 2**expo < upper_lim:
		K[(2**expo)-1] = expo+1
		expo += 1

	for  i in range(1, upper_lim+1, 1):
		print i
		memo_seed_length(i, K)
	return K.index(max(K))+1

if __name__ == '__main__':
	upper_lim = 1000000

	# MEMOIZATION
	print "Result ",bottom_up_memoization(upper_lim)
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Project Euler - Problem 12](http://www.erogol.com/project-euler-problem-12/ "Project Euler - Problem 12")
2. [Project Euler - Problem 13](http://www.erogol.com/project-euler-problem-13/ "Project Euler - Problem 13")
3. [de-importing a Python module with a simple function](http://www.erogol.com/de-importing-python-module-simple-function/ "de-importing a Python module with a simple function")