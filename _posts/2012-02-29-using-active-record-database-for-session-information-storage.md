---
layout: post
title: "Using Active Record for Session Storage"
description: "If you have a problem about the cookie size (4KB) in your application, you may use database storage "
tags: 
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

If you have a problem about the cookie size (4KB) in your application, you may use database storage help. With this manipulation your app will keep an id in cookie and store other information in “sessions” table in DB and it will call the data from table by the id that is kept in cookie.

**First**: Execute following (it will create the table structure that is ready to be migrated)

rake db:sessions:create

**Second**: It’ll create the table

rake db:migrate

**Third**: inside the config/initializers/session\_store.rb add following

MyApp::Application.config.session\_store :active\_record\_store

That s all folks…

Enjoy:)

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.