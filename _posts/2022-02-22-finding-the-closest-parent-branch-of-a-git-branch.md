---
layout: post
title: Finding the closest parent of a git branch.
description: Sample code
tags: coding
minute: 0.00001
---

```terminal
git show-branch \
| sed "s/].*//" \
| grep "\*" \
| grep -v "$(git rev-parse --abbrev-ref HEAD)" \
| head -n1 \
| sed "s/^.*\[//"
```