---
layout: post
title: "Understanding Rails Authenticity Token"
description: "What happens:  
When the user views a form to create, update, or destroy a resource, the rails app w"
tags: ruby ruby_on_rails
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

What happens:  
When the user views a form to create, update, or destroy a resource, the rails app would create a random authenticity\_token, store this token in the session, and place it in a hidden field in the form. When the user submits the form, rails would look for the authenticity\_token, compare it to the one stored in the session, and if they match the request is allowed to continue.

Why this happens:  
Since the authenticity token is stored in the session, the client can not know its value. This prevents people from submitting forms to a rails app without viewing the form within that app itself. Imagine that you are using service A, you logged into the service and everything is ok. Now imagine that you went to use service B, and you saw a picture you like, and pressed on the picture to view a larger size of it. Now, if some evil code was there at service B, it might send a request to service A (which you are logged into), and ask to delete your account, by sending a request to http://serviceA.com/close\_account. This is what is known as CSRF (Cross Site Request Forgery).

If service A is using authenticity tokens, this attack vector is no longer applicable, since the request from service B would not contain the correct authenticity token, and will not be allowed to continue.

Notes: Keep in mind, rails only checks POST, PUT, and DELETE requests. GET request are not checked for authenticity token. Why? because the HTTP specification states that GET requests should NOT create, alter, or destroy resources at the server, and the request should be idempotent (if you run the same command multiple times, you should get the same result every time).

Lessons: Use authenticity\_token to protect your POST, PUT, and DELETE requests. Also make sure not to make any GET requests that could potentially modify resources on the server.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.