---
layout: post
title: "What I Learned Today: Database Sharding"
description: "Sharding database consept was founded before a decade however implementations have occurred with com"
tags: database
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Sharding database consept was founded before a decade however implementations have occurred with complex social networking apps and SaaS and actually it was coined by Google pratically.

Database Sharding can be simply defined as a “shared-nothing” partitioning scheme for large databases across a number of servers, enabling new levels of database performance and scalability achievable. If you think of broken glass, you can get the concept of sharding – breaking your database down into smaller chunks called “shards” and spreading those across a number of distributed servers.

The basic concept of Database Sharding is very straightforward: take a large database, and break it into a number of smaller databases across servers. The concept is illustrated in the following diagram:

![Database Sharding](https://www.codefutures.com/img/dbshards-shardit.gif)

Figure 2. Database Sharding takes large databases and breaks them down into smaller databases.

The obvious advantage of the shared-nothing Database Sharding approach is improved scalability, growing in a near-linear fashion as more servers are added to the network. However, there are several other advantages of smaller databases, which should not be overlooked when considering a sharding solution:

* *Smaller databases are easier to manage.* Production databases must be fully managed for regular backups, database optimization and other common tasks. With a single large database these routine tasks can be very difficult to accomplish, if only in terms of the time window required for completion. Routine table and index optimizations can stretch to hours or days, in some cases making regular maintenance infeasible. By using the sharding approach, each individual “shard” can be maintained independently, providing a far more manageable scenario, performing such maintenance tasks in parallel.
* *Smaller databases are faster.* The scalability of sharding is apparent, achieved through the distribution of processing across multiple shards and servers in the network. What is less apparent is the fact that each individual shard database will outperform a single large database due to its smaller size. By hosting each shard database on its own server, the ratio between memory and data on disk is greatly improved, thereby reducing disk I/O. This results in less contention for resources, greater join performance, faster index searches, and fewer database locks. Therefore, not only can a sharded system scale to new levels of capacity, individual transaction performance is benefited as well.
* *Database Sharding can reduce costs.* Most Database Sharding implementations take advantage of lower-cost open source databases, or can even take advantage of “workgroup” versions of commercial databases. Additionally, sharding works well with commodity multi-core server hardware, far less expensive than high-end multi-CPU servers and expensive SANs. The overall reduction in cost due to savings in license fees, software maintenance and hardware investment is substantial, in some cases 70% or more when compared to other solutions.

There is no doubt that Database Sharding is a viable solution for many organizations, supported by the number of large online vendors and SaaS organizations that have implemented the technology (giants such as Amazon, eBay, and of course Google).

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.