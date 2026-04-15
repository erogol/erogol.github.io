---
layout: post
title: "Using Yield in Ruby"
description: "yield function is one of the good stuff in the Ruby that gives different kind of code reuse logic to"
tags: coding example ruby
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

yield function is one of the good stuff in the Ruby that gives different kind of code reuse logic to your code. It just like lambda in Scheme and Lips. when yield is called in a function it cuts the execution of its container function and pass the execution time to the do-end block then when the execution ends with the end keyword, it continues the execution from the last point.

Here is an example code that uses yield function.

```python
def around_staff
  eren = 20
  puts "first step"
  yield(eren)
  puts "last step"
end

def do_something
  around_staff do |eren|
    puts "I did something around"+ eren.to_s
  end
end

do_something
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Setting up .sort function to your custom class in RUBY](https://erogol.com/setting-up-sort-function-to-your-custom-class-in-ruby/ "Setting up .sort function to your custom class in RUBY")