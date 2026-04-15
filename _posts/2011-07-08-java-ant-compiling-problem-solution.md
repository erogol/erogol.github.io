---
layout: post
title: "Java Ant Compiling Problem Solution"
description: "I was working on a Strunt2 framework project and I needed to use ant in eclipse"
tags: ant compile java problem
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I was working on a Strunt2 framework project and I needed to use ant in eclipse. I got a problem like this:

```python
BUILD FAILED: D:workspacespschemabuild.xml:37:
Unable to find a javac compiler;
com.sun.tools.javac.Main is not on the classpath.
Perhaps JAVA_HOME does not point to the JDK
```

The solution is to go Window>>Preferences>>Ant>>Runtime than add the "tools.jar" file (somewhere in the JDK's folder) to jars that are in the Heading "Ant Home Entries".

That's it 🙂 have fun!

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.