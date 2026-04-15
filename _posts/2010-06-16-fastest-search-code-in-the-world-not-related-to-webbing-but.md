---
layout: post
title: "Fastest Search Code In The World Not Related To Webbing But"
description: "This is implementation of the radix sort in binary base with C++ code"
tags: 
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**This is implementation of the radix sort in binary base with C++ code. You could see the O(1) implementation of radix sort with huge fast difference from other algorithms.**

void radixAt(int numberOfItems,int byteNumber, int \*temp, int \*arr){  
 int count[256];  
 int index[256];  
 for(int i = 0; i<256; i++){//fill all elements of count array with 0  
 count[i] =0;  
 }  
 for(int i = 0; i < numberOfItems; i++){//determine the number of each byte amounth  
 count[(arr[i] >> (byteNumber\*8))&0xff]++;//take byte to controll  
 }  
 index[0] = 0;  
 for(int i = 1; i < 256 ; i++){//for each byte amounth, keep the index space   
 index[i] = index[i-1] + count[i-1];  
 }  
 for(int i = 0; i < numberOfItems ; i++){//fill elements according to their byte to temp array for next call to main array  
 temp[index[(arr[i]>>(byteNumber\*8))&0xff]++] = arr[i];  
 }  
}  
void radixSort (int \*arr, int \*tempArr, int size){  
 radixAt(size, 0, tempArr, arr);  
 radixAt(size, 1, arr, tempArr);  
 radixAt(size, 3, tempArr, arr);  
 radixAt(size, 4, arr, tempArr);  
}

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.