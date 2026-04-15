---
layout: post
title: "Setting Up Selenium on RaspberryPi 2/3"
description: "Selenium is a great tool for Internet scraping or automated testing for websites"
tags: firefox howto installation python selenium
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Selenium is a great tool for Internet scraping or automated testing for websites. I personally use it for scrapping on dynamic content website in which the content is created by JavaScript routines. Lately, I also tried to run Selenium on Raspberry and found out that it is not easy to install all requirements. Here I like to share my commands to make things easier for you.

Here I like to give a simple run-down to install all requirements to make Selenium available on a Raspi. Basically, we install first Firefox, then Geckodriver and finally Selenium and we are ready to go.

Before start,  better to note that ChromeDriver does not support ARM processors anymore, therefore it is not possible to use Chromium with Selenium on Raspberry.

First, install system requirements. Update the system, install Firefox and xvfb (display server implementing X11);

```python
sudo apt-get update
sudo apt-get install iceweasel
sudo apt-get install xvfb
```

Then, install python requirements. Selenium, PyVirtualDisplay that you can use for running Selenium with hidden  browser display and xvfbwrapper.

```python
sudo pip install selenium
sudo pip install PyVirtualDisplay
sudo pip install xvfbwrapper
```

Hope everything run well and now you can test the installation.

```python
from pyvirtualdisplay import Display
from selenium import webdriver

display = Display(visible=0, size=(1024, 768))
display.start()

driver = webdriver.Firefox()
driver.get('http://www.erogol.com/')
driver.quit()

display.stop()
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [How to use Python Decorators](http://www.erogol.com/use-python-decorators/ "How to use Python Decorators")
2. [Parallelized Machine Learning with Python and Sklearn](http://www.erogol.com/parallelized-machine-learning-python-sklearn/ "Parallelized Machine Learning with Python and Sklearn")
3. [Simple Parallel Processing in Python](http://www.erogol.com/simple-parallel-processing-python/ "Simple Parallel Processing in Python")
4. [Passing multiple arguments for Python multiprocessing.pool](http://www.erogol.com/passing-multiple-arguments-python-multiprocessing-pool/ "Passing multiple arguments for Python multiprocessing.pool")