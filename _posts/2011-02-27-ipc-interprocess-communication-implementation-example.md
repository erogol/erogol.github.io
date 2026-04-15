---
layout: post
title: "IPC Interprocess Communication Implementation Example"
description: "This example code illustrates the basic implementation of IPC mechanism that is provided by POSIX AP"
tags: example ipc linux posix
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

This example code illustrates the basic implementation of IPC mechanism that is provided by POSIX API.

`#include   
#include shm.h>  
>#include stat.h>  
>  
int main(int argc, char **argv)  
{`

//create shared mem.  
 segment\_id = shmget(IPC\_PRIVATE, size, S\_IRUSR|S\_IWUSR);

//attach the shared mem. to the process in Write Read mode.(0 argument)  
 shared\_mem = (char \* )shmat(segment\_id, NULL, 0);

//write the data to the shared memory.  
 sprintf(shared\_mem, "This is the shared memory example coded by Eren Golge. Regards!!");

//read the data from  
 printf("\*%s n", shared\_mem );

//detach the segment from process  
 shmdt(shared\_mem);

//delete the shared mem. segment form memory  
 shmctl(segment\_id, IPC\_RMID, NULL);

printf("End of code");  
 return 0;  
}

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.