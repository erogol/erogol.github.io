---
layout: post
title: "What I learnt about Ruby and Rails today?"
description: "Difference beween attr\_accessor and attr\_accesible:

attr\_accessor is a ruby method that makes a "
tags: ruby rubyandrails
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Difference beween attr\_accessor and attr\_accesible:**

attr\_accessor is a ruby method that makes a getter and a setter. attr\_accessible is a Rails method that allows you to pass in values to a mass assignment: new(attrs) or up update\_attributes(attrs).

Here's a mass assignment:

```python
Order.new({ :type => 'Corn', :quantity => 6 })
```

You can imagine that the order might also have a discount code, say :price\_off. If you don't tag :price\_off as attr\_accessible you stop malicious code from being able to do like so:

```python
Order.new({ :type => 'Corn', :quantity => 6, :price_off => 30 })
```

Even if your form doesn't have a field for :price\_off, if it's just in your model by default it's available so a crafted POST could still set it. Using attr\_accessible white lists those things are can be mass assigned.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.