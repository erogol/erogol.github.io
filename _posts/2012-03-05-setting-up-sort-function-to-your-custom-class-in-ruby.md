---
layout: post
title: "Setting Up Custom Sort in Ruby"
description: "class Eren  
include Comparable  
attr_accessor :age  
 def initialize _age  
@age = _age  
end  
 d"
tags: coding example ruby
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

`class Eren  
include Comparable  
attr_accessor :age`  
 `def initialize _age  
@age = _age  
end`  
 `def <=> other  
self.age <=> other.age  
end  
end`  
 `e1 = Eren.new(10)  
e2 = Eren.new(11)  
e3 = Eren.new(2)`  
 `d = [e1,e2,e3].sort!  
d.each do |x|  
puts x.age  
end`

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Use yield in Ruby](https://erogol.com/use-yield-in-ruby/ "Use yield in Ruby")