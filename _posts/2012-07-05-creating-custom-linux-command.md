---
layout: post
title: "Creating Custom Linux Command"
description: "Sometimes you may need to repeat lots of linux command on terminal for any aim"
tags: command create linux
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Sometimes you may need to repeat lots of linux command on terminal for any aim. It is some time so boring to type all the commands again and again so the solution to create  own commands to execute all the repetition in one.

All the commands are kept on /bin directory s we just need to create a file and name as the command that we’ll use. Then we need to make the file readable and executable.

For example I create a apache server restart command.

– Go to /bin path: **cd /bin**

– Create the command file with the command name:**sudo touch apacherestart**

– Make the fie readable and executable: **sudo chmod +rx apacherestart**

– Open the file and write the command we’ll use the new one instead of it: **sudo /etc/init.d/apache2 restart**

That’s all. After now, we might use the new command to restart apache server: **apacherestart**

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Today's Linux Console Command](https://erogol.com/todays-linux-console-command/ "Today's Linux Console Command")
2. [A bash script for Linux to extract all kind of compressed file from console.](https://erogol.com/a-bash-script-for-linux-to-extract-all-kind-of-compressed-file-from-console/ "A bash script for Linux to extract all kind of compressed file from console.")