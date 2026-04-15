---
layout: post
title: "What is Asynchronous Programming?"
description: "!Thread Execution(https://krondo"
tags: asynchronous coding programming thread
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

![Thread Execution](https://krondo.com/blog/wp-content/uploads/2009/08/threaded.png "Normal Thread Execution")
:   Normal Thread Execution

![asynchtomous execution](https://krondo.com/blog/wp-content/uploads/2009/08/async.png "asynchtomous execution")

asynchtomous execution

Start with the comparison (that assumes you know normal threaded execution).

Two main difference between normal threaded system and asynchronous system are:

* For threaded execution each thread has its own controller, however for asynchronous system there is only one thread controller.
* Threaded execution does not give the control of ending, starting, changing to user. It is mainly controlled by the operating system internals. On the other side asynchronous execution need some explicit command to interleave one execution to other. It is more in control in the programmer’s perspective.

The reason behind using asynchronous is it makes your code drastically more efficient for the large number of tasks that are less communicative with each other and have plenty number of I/O operations or any other operation force them to wait for a little.

![asynchronous execution vs task waiting](https://krondo.com/blog/wp-content/uploads/2009/08/block.png "asynchronous execution vs task waiting")

asynchronous execution vs task waiting

As you can see from the left figure , there are three tasks follow each other. The grey fields are symbolizing the some waiting fragment in time because of I/O operations (like downloading, printing, scanning). So as you it can be seen from the figure the waste of time is really large in that normal sequential execution.

Thus with asynchronous execution you might deal with these time wastes and instead of waiting some, you can go with the another process. However you may face with some new difficulties. For example there can be some data need to be interchanged between tasks. Now you need to also consider the sequence of executions. This is the one of the main concern of asynchronous execution.

At last, these are some underpinning conditions that makes asynchronous execution viable:

1. There are a large number of tasks so there is likely always at least one task that can make progress.
2. The tasks perform lots of I/O, causing a synchronous program to waste lots of time blocking when other tasks could be running.
3. The tasks are largely independent from one another so there is little need for inter-task communication (and thus for one task to wait upon another).

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.