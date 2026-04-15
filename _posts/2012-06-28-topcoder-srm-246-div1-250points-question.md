---
layout: post
title: "Topcoder SRM 246 Div1 250points question"
description: "My silly solution for that question:

Problem Statement

The problem statement contains the unicode "
tags: div1 solution srm top_coder
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

My silly solution for that question:

Problem Statement

The problem statement contains the unicode symbols.  
 You are developing a new software calculator. A very important feature is the auto-placing of the ? value by one click. The only problem is that you don't know the required precision. That's why you decided to write a program that can return ? with any reasonable precision.  
 You are given an int precision. You should return the ? value with exactly precision digits after the decimal point. The last digit(s) should be rounded according to the standard rounding rules (less than five round down, more than or equal to five round up).  
 Definition

Class:  
 PiCalculator  
 Method:  
 calculate  
 Parameters:  
 int  
 Returns:  
 String  
 Method signature:  
 String calculate(int precision)  
 (be sure your method is public)

Notes  
 -  
 ? equals 3.141592653589793238462643383279...  
 Constraints  
 -  
 precision will be between 1 and 25, inclusive.  
 Examples  
 0)

2  
 Returns: "3.14"

1)

4  
 Returns: "3.1416"  
 The value should be rounded.  
 2)

12  
 Returns: "3.141592653590"  
 Be careful with rounding.

```python
public class PiCalculator{
	public String calculate(int precision){
		String value = "3.141592653589793238462643383279";
		String prec = value.substring(0,precision+2);

		int val_int = Integer.valueOf(value.substring(prec.length(),prec.length()+1));

	    int is_9 = 0; //1 is true 0 is false;

	   	if(val_int >= 5){
	   	     int flag = prec.length();
	   	     do{
	    	   String prec_last = prec.substring(flag-1,flag);
	    	   int prec_last_int = Integer.valueOf(prec_last);
	    	  System.out.println(prec_last_int);
	    	   if(prec_last_int == 9){
	    	   		is_9 = 1;
	    	   		prec_last_int = 0;
	    	   		flag--;
	    	   		prec = prec.substring(0,flag)+(prec_last_int+"");
	    	   		//System.out.println(prec);
	    	   }else{
	    	   		prec_last_int++;
	    	   		is_9 = 0;
	    	   		prec = prec.substring(0,flag-1)+(prec_last_int+"")+prec.substring(flag,prec.length());
	    	   		//System.out.println(prec);
	    	   }
	   		 }while(is_9 == 1);
		}

		return prec;
	}
}
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.