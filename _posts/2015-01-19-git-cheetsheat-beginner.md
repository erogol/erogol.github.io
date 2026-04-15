---
layout: post
title: "Git CheetSheat for a beginner"
description: "!LinkedIn(https://web"
tags: cheat sheet git
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

var disqus\_url = 'http://www.erogol.com/git-cheetsheat-beginner/';
var disqus\_identifier = '1100 http://www.erogol.com/?p=1100';
var disqus\_container\_id = 'disqus\_thread';
var disqus\_shortname = 'erogol';
var disqus\_title = "Git CheetSheat for a beginner";
var disqus\_config\_custom = window.disqus\_config;
var disqus\_config = function () {
/\*
All currently supported events:
onReady: fires when everything is ready,
onNewComment: fires when a new comment is posted,
onIdentify: fires when user is authenticated
\*/
this.language = '';
this.callbacks.onReady.push(function () {
// sync comments in the background so we don't block the page
var script = document.createElement('script');
script.async = true;
script.src = '?cf\_action=sync\_comments&post\_id=1100';
var firstScript = document.getElementsByTagName('script')[0];
firstScript.parentNode.insertBefore(script, firstScript);
});
if (disqus\_config\_custom) {
disqus\_config\_custom.call(this);
}
};
(function() {
var dsq = document.createElement('script'); dsq.type = 'text/javascript';
dsq.async = true;
dsq.src = '//' + disqus\_shortname + '.disqus.com/embed.js';
(document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(dsq);
})();