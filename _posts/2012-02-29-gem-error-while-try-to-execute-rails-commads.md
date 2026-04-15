---
layout: post
title: "Gem Error While Try To Execute Rails Commads"
description: "to_specs': Could not find rails (>= 0) amongst  (Gem::LoadError) from /usr/local/lib/site_ruby/1"
tags: error solution
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

`to_specs': Could not find rails (>= 0) amongst [] (Gem::LoadError) from /usr/local/lib/site_ruby/1.8/rubygems/dependency.rb:256:in`

if you get such an error while execution “rails new [project\_name]” it means there is a conflict on your machine between gem installations and you need to get rid one of them.

I just start to use rvm instead of plain ruby installation and there were an such error. As a solution I deleted the “site\_ruby” folder in ~/opt/local/lib path, then everything is working fine

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.