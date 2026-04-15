---
layout: post
title: "What is \"long long\" type in c++?"
description: "long long is not the same as long (although they can have the same size, e"
tags: c c programming tutorial
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

`long long` is not the same as `long` (although they can have the same size, e.g. in most [64-bit POSIX system](http://en.wikipedia.org/wiki/64-bit#Specific_C-language_data_models)). It is just guaranteed that a `long long` is at least as long as a `long`. In most platforms, a `long long` represents a 64-bit signed integer type.

You could use `long long` to store the 8-byte value safely in most conventional platforms, but it's better to use `int64_t`/`int_least64_t` from `<stdint.h>`/`<cstdint>` to clarify that you want an integer type having ≥64-bit.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Getting started to Thrust on source code...](http://www.erogol.com/getting-started-to-thrust-on-soruce-code/ "Getting started to Thrust on source code...")
2. [Sorting strings and Overriding std::sort comparison](http://www.erogol.com/sorting-strings-overriding-stdsort-comparison/ "Sorting strings and Overriding std::sort comparison")
3. [Some possible Matrix Algebra libraries based on C/C++](http://www.erogol.com/some-possible-matrix-algebra-libraries-based-on-cc/ "Some possible Matrix Algebra libraries based on C/C++")
4. [Extracting a sub-vector at C++](http://www.erogol.com/extracting-sub-vector-c/ "Extracting a sub-vector at C++")