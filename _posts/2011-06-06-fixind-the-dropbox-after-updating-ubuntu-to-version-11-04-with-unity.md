---
layout: post
title: "Fixind the DropBox after updating Ubuntu to version 11.04 with Unity."
description: "Those of you that run Ubuntu 11"
tags: dropbox fix ubuntu unity
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Those of you that run Ubuntu 11.04 Natty Narwhal already probably know that the Dropbox application indicator doesn't work in Natty - not using the official Dropbox build.

But you can get the Dropbox AppIndicator to work in Ubuntu 11.04 Natty Narwhal by using the latest script below:

1. Install Dropbox the regular way - using the .deb provided on its download page.

2. Run the commands below:

```python
cd
wget http://webupd8.googlecode.com/files/fixdropbox
chmod +x fixdropbox
./fixdropbox
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.