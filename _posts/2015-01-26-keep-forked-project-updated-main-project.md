---
layout: post
title: "How to keep your forked project updated with the main project ?"
description: "python


 Add the remote, call it "upstream":

git remote add upstream https://github"
tags: cheat sheet github
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

```python


# Add the remote, call it "upstream":

git remote add upstream https://github.com/whoever/whatever.git

# Fetch all the branches of that remote into remote-tracking branches,
# such as upstream/master:

git fetch upstream

# Make sure that you're on your master branch:

git checkout master

# Rewrite your master branch so that any commits of yours that
# aren't already in upstream/master are replayed on top of that
# other branch:

git rebase upstream/master

#If you don't want to rewrite the history of your master branch, (for # example because other people may have cloned it) then you should # replace the last command with However, for making further pull    # requests that are as clean as possible, it's probably better to # rebase.
git merge upstream/master. 

```
