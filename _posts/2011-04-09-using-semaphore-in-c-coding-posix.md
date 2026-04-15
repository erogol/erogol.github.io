---
layout: post
title: "Using Semaphore in C coding (Posix)"
description: "Semaphore is used for dealing with the synchronisation problem between processes and threads"
tags: c posix semaphore thread
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Semaphore is used for dealing with the synchronisation problem between processes and threads. It is well defined library structure that comes with the POSIX libraries.

Here a example that have a thread synchronisation problem between two threads. Here is the code.

`#include <pthread.h>  
#include <stdio.h>  
#include <stdlib.h>  
#define UPTO 10000000  
int count = 0;  
void * ThreadAdd(void * a)  
{  
int i, tmp;  
for(i = 0; i < UPTO; i++)  
{  
sem_wait(&mutex);  
tmp = count;      /* copy the global count locally */  
tmp = tmp+1;      /* increment the local copy */  
count = tmp;      /* store the local value into the global count */  
//printf("The count is %d in %dn", count, (int)pthread_self());  
sem_post(&mutex);  
}  
}  
int main(int argc, char * argv[])  
{  
pthread_t tid1, tid2;  
if(pthread_create(&tid1, NULL, ThreadAdd, NULL))  
{  
printf("n ERROR creating thread 1");  
exit(1);  
}  
if(pthread_create(&tid2, NULL, ThreadAdd, NULL))  
{  
printf("n ERROR creating thread 2");  
exit(1);  
}  
if(pthread_join(tid1, NULL))    /* wait for the thread 1 to finish */  
{  
printf("n ERROR joining thread");  
exit(1);  
}  
if(pthread_join(tid2, NULL))        /* wait for the thread 2 to finish */  
{  
printf("n ERROR joining thread");  
exit(1);  
}  
if (count < 2 * UPTO)  
printf("n BOOM! count is [%d], should be %dn", count, 2*UPTO);  
else  
printf("n OK! count is [%d]n", count);  
pthread_exit(NULL);  
}`

This code will have the synchronisation problem when you use it since the shared variable count will be updated with two threads unexpectedly. Consider the "for" section of the function "ThreadAdd". In that section count variable is updated but this increment and update action is not atomic operation (that means, it needs multiple system calls in low level) so in the middle of the for loop the other thread can get the control and continue its execution. As a result it causes unexpected result. Let's analysis execution flow of the threads.

initially "count = 1"

**Thread 1:**  
tmp = count;  
tmp = tmp+1  
Thread2  
Thread2  
Thread2  
count = tmp //(count = 2)

**Thread 2:**  
Thread1  
Thread1  
tmp = count;  
tmp = tmp+1  
count = tmp  //(count = 2)  Two for loops the give same result for count. So result is 2 instead of 3

For this kind of problems we can easily use Semaphore. Here the basic functions:

**sem\_t sem\_name;** //create variable

**sem\_init(&sem\_name, Flag, Init\_val)** // initialise the semaphore var. "Flag" decides whether it is shared by processes. "Init\_val" is the initial value of the semaphore.

**sem\_wait(&sem\_name);**//waits while semaphore value will  be positive, if value is positive it makes the semaphore negative again to make he other processes waiting.

**sem\_post(&sem\_name);**//increment the semaphore value, so one of the waiting thread can continue its operations.

**sem\_destroy(&sem\_name);**

Here the solution of the above code with semaphore:

`;  
#include <pthread.h>  
#include <unistd.h>  
#include <sys/types.h>  
#include <semaphore.h>  
#include <stdio.h>  
#include <stdlib.h>  
#define UPTO 10000000  
int count = 0;  
sem_t mutex;  
void * ThreadAdd(void * a)  
{  
int i, tmp;  
for(i = 0; i < UPTO; i++)  
{  
sem_wait(&mutex);  
tmp = count;      /* copy the global count locally */  
tmp = tmp+1;      /* increment the local copy */  
count = tmp;      /* store the local value into the global count */  
//printf("The count is %d in %dn", count, (int)pthread_self());  
sem_post(&mutex);  
}  
}  
int main(int argc, char * argv[])  
{  
pthread_t tid1, tid2;  
sem_init(&mutex, 0, 1);  
if(pthread_create(&tid1, NULL, ThreadAdd, NULL))  
{  
printf("n ERROR creating thread 1");  
exit(1);  
}  
if(pthread_create(&tid2, NULL, ThreadAdd, NULL))  
{  
printf("n ERROR creating thread 2");  
exit(1);  
}  
if(pthread_join(tid1, NULL))    /* wait for the thread 1 to finish */  
{  
printf("n ERROR joining thread");  
exit(1);  
}  
if(pthread_join(tid2, NULL))        /* wait for the thread 2 to finish */  
{  
printf("n ERROR joining thread");  
exit(1);  
}  
if (count < 2 * UPTO)  
printf("n BOOM! count is [%d], should be %dn", count, 2*UPTO);  
else  
printf("n OK! count is [%d]n", count);  
sem_destroy(&mutex);  
pthread_exit(NULL);  
}`

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Using Pipes for IPC in C](http://www.erogol.com/using-pipes-for-ipc-in-c/ "Using Pipes for IPC in C")
2. [Convertion the integer to string in C!!](http://www.erogol.com/convertion-the-integer-to-string-in-c/ "Convertion the integer to string in C!!")
3. [Extracting a sub-vector at C++](http://www.erogol.com/extracting-sub-vector-c/ "Extracting a sub-vector at C++")
4. [Sorting strings and Overriding std::sort comparison](http://www.erogol.com/sorting-strings-overriding-stdsort-comparison/ "Sorting strings and Overriding std::sort comparison")