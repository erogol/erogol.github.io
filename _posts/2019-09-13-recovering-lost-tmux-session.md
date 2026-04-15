---
layout: post
title: "Recovering Lost Tmux Session"
description: "After a while of using tmux, you might see that you cannot reconnect it from another terminal window"
tags: session tmux
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

After a while of using tmux, you might see that you cannot reconnect it from another terminal windows with the error message

`error connecting to /tmp/tmux-1000/default (No such file or directory`

The solution is easy but hard to find. Here is the magical command worked for me. Hope it works for you too!

```python
pkill -USR1 tmux
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.