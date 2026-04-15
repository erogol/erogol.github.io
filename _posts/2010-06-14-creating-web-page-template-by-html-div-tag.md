---
layout: post
title: "Creating Web Page Template By Html Div Tag"
description: "You need to know CSS and HTML for this topic"
tags: 
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

You need to know CSS and HTML for this topic.

Now it is different piece from JavaScript but it is fundamental side of web page design. Creating a template makes your life easy while you are designing a web page since, you can easily shape your pages by adding some necessaries to template.

Today’s most common way to create a template is using div tag. Basically you define a container “division” that is the main division that includes the other container inside it. Then we add others respected to our design to its inside. Do not forget to give id or class name to each division to be specified for CSS. (Like as in the figure.)

[![](https://1.bp.blogspot.com/_LrqoV55zTWE/TBYBQW5zR9I/AAAAAAAAABQ/IuwGQWpnWHY/s320/division-diagram.gif)](http://1.bp.blogspot.com/_LrqoV55zTWE/TBYBQW5zR9I/AAAAAAAAABQ/IuwGQWpnWHY/s1600/division-diagram.gif)

```python
Example HTML for such template;
```

But it is not enough to have this HTML code, also we need to use some CSS to define our divisions’ width and height.

```python
CSS code  
.container{  
 width: 960px;  
 background-color:#FFF;  
 height: 500px;  
 margin-left:auto;  
 margin-right:auto;  
}  
  
.left{  
 float: left;  
 width: 430px;  
 height: 490px;  
 background-color: #DFDFDF;  
 padding: 10px;  
 overflow:auto;  
}  
  
.right{  
 float: left;  
 width: 430px;  
 height: 490px;  
 background-color: #BFBFBF;  
 padding: 10px;  
 overflow:auto;  
}
```

[To see the resulted page click!](http://www.erengolge.0fees.net/WebPageTemplateExp.html)

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.