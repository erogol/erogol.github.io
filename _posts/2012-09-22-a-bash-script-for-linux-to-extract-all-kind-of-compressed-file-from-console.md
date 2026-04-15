---
layout: post
title: "A Bash Script For Linux To Extract All Kind Of Compressed File From Console"
description: "Open ~/"
tags: command console linux
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Open ~/.bashrc file with your favorite editor and paste below script to the bottom of the file.

```python
extract () {
if [ -f ![1 ] ; then     case](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-be4088a43f862256ccd848e75b87c279_l3.svg "Rendered by QuickLaTeX.com")1 in
     *.tar.bz2) tar xvjf ![1 ;;      *.tar.gz) tar xvzf](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-474fbe278f9d6b0d7db6e5b32fed59b9_l3.svg "Rendered by QuickLaTeX.com")
```

then write “extract” on console with the file you want to extract and see the effect.

```python
extract bla.tar.gz
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Today's Linux Console Command](https://erogol.com/todays-linux-console-command/ "Today's Linux Console Command")
2. [Creating Custom Linux Command](https://erogol.com/creating-custom-linux-command/ "Creating Custom Linux Command")