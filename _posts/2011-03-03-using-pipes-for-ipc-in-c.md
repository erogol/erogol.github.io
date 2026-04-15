---
layout: post
title: "Using Pipes for IPC in C"
description: "I am developing a program that utilises the pipe system of POSIX for my Operating system course"
tags: c ipc pipe posix
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I am developing a program that utilises the pipe system of POSIX for my Operating system course. I want to share my experience with you. There might be some flaws in my explanations and in that case please WARN ME?

Then, lets start some explanation for beginning. Pipes system is used for (Inter-process Communication) IPC between related processes. Related means that the processes having parent-child relation or sibling relation. It is not possible to use pipes for unrelated process communication. In addition the another limitation of the pipes is that one pipe just gives uni-directional communication between processes so if you want to have both direction communication, you might use two distinct pipes that have different directions in the way of sharing data (one pipe to send data form parent to child, one pipe to send data from child to parent). Another restriction of the pipe system, it is not the best way to share big data between processes (Using shared memory structure may be suitable) but it is the most common way.

Now here the basic process of coding in C:

First #include stdio.h(standard input output library) stdlib.h (standard library) unistd.h(POSIX depended functions library)

Create the pipe with "pipe(fd)". fd is the array to keep the file descriptors that are returned by the function. These descriptors points the "files" that are created after the function to provide read and write. (By the way, pipes are respected as files in the system. Thus each read and write action manipulated like file objects). Here is the code:  
 `int fd[2];  
pipe(fd);`

After you create the pipe, if you "fork()" the process, it creates a child process that inherits all the information about the pipe so it is okay to communicate with the pipes.

I give an example code taken and little manipulated by myself form a pdf. In this implementation, by using the file descriptors we create file objects. In that way, this is possible to use standard C functions to read and write pipes. (Also you can get the PDF file from [here](http://www.mediafire.com/?glwa4fg2jadbjdp)).  
 [Here](http://www.mediafire.com/?970d4iicil4lu1e) the link for below code:

#include   
 #include   
 #include   
 #include

/\* Read characters from the pipe and echo them to stdout. \*/

void  
 read\_from\_pipe (int file)  
 {  
 FILE \*stream;  
 int c;  
 stream = fdopen (file, "r");  
 while ((c = fgetc (stream)) != EOF)  
 putchar (c);  
 fclose (stream);  
 }

/\* Write some random text to the pipe. \*/

void  
 write\_to\_pipe (int file)  
 {  
 FILE \*stream;  
 stream = fdopen (file, "w");  
 fprintf (stream, "hello, world!n");  
 fprintf (stream, "goodbye, world!n");  
 fclose (stream);  
 }

int  
 main (void)  
 {  
 pid\_t pid;  
 int mypipe[2];

/\* Create the pipe. \*/  
 if (pipe (mypipe))  
 {  
 fprintf (stderr, "Pipe failed.n");  
 return EXIT\_FAILURE;  
 }

/\* Create the child process. \*/  
 pid = fork ();  
 if (pid == (pid\_t) 0)  
 {  
 /\* This is the child process.  
 Close other end first. \*/  
 close (mypipe[1]);  
 read\_from\_pipe (mypipe[0]);  
 return EXIT\_SUCCESS;  
 }  
 else if (pid < (pid\_t) 0) {   
/\* The fork failed. \*/   
fprintf (stderr, "Fork failed.n");   
return EXIT\_FAILURE; }   
else { /\* This is the parent process. Close other end first. \*/   
close (mypipe[0]);   
write\_to\_pipe (mypipe[1]);   
return EXIT\_SUCCESS;   
}   
}   
  
Also there is another way of using pipes by using "popen" as named pipe implementation. It is more generalised way of doing and avoids all the file object creation. You can get more info from the pdf.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Fine document for IPC alternatives.](http://www.erogol.com/fine-document-for-ipc-alternatives/ "Fine document for IPC alternatives.")
2. [IPC (Interprocess Communication) implementation example.](http://www.erogol.com/ipc-interprocess-communication-implementation-example/ "IPC (Interprocess Communication) implementation example.")
3. [Using Semaphore in C coding (Posix)](http://www.erogol.com/using-semaphore-in-c-coding-posix/ "Using Semaphore in C coding (Posix)")
4. [What is "long long" type in c++?](http://www.erogol.com/what-is-long-long-type-in-c/ "What is \"long long\" type in c++?")