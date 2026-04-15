---
layout: post
title: "Microsoft Research introduced a new NN model that beats Google and the others"
description: "MS researcher recently introduced a new deep ( indeed very deep 🙂 ) NN model (PReLU Net) 1 and they "
tags: deep learning imagenet machine learning neural network prelu
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

MS researcher recently introduced a new deep ( indeed very deep 🙂 ) NN model (PReLU Net) [1] and they push the state of art in ImageNet 2012 dataset from 6.66% (GoogLeNet) to 4.94% top-5 error rate.

In this work, they introduce an alternation of well-known ReLU activation function. They call it PReLu (Parametric Rectifier Linear Unit). The idea behind is to allow negative activations on the ReLU function with a control parameter ![a](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_0cc175b9c0f1b6a831c399e269772661.gif) which is also learned over the training phase. Therefore, PReLU allows negative activations and in the paper they argue and emprically show that PReLU is better to resolve diminishing gradient problem for very deep neural networks  (> 13 layers) due to allowance of negative activations. That means more activations per layer, hence more gradient feedback at the backpropagation stage.

[![PReLU](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU.jpg)

all figures are from the paper

As I told earlier, PReLU requires a new learned parameter ![a](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_0cc175b9c0f1b6a831c399e269772661.gif) for each unit (channel-wise) or for each layer (channel-shared). In both cases, they show that PReLU increases the results as oppose to ReLU activations for especially deeper models. For being more precise lets dive into formulations. PReLU behaves with the following function for the feedforward propagation;

[![PReLU_act](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_act.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_act.jpg)

![a](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_0cc175b9c0f1b6a831c399e269772661.gif) is updated for each epoch with the following formulation. Compute gradient which is the gradient of the deeper layer multiplied by the layer unit stimuli ![y](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_415290769594460e2e485922904f345d.gif), if unit activation ![f(y) > 0](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_7a0e0b0be9b8b7986ccc443fad35b8b0.gif). Gradient is ![0](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_cfcd208495d565ef66e7dff9f98764da.gif) otherwise.

[![PReLU_bp](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp.jpg)[![PReLU_bp2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp2.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp2.jpg)[![PReLU_bp3](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp3.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU_bp3.jpg)

![mu](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_89aa4b196b48c8a13a6549bb1eaebd80.gif) is momemtum, ![epsilon](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_3cd38ab30e1e7002d239dd1a75a6dfa8.gif) is learning rate.

Beside the PReLU function, they also use Spatial Pyramid Pooling  (SPP) layer [2] just before the fully connected layers. Being a side note, SPP is a great tool that makes you able to process different size images and evades the size constraint of the NN models.

[![PReLU2](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU2.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2015/02/PReLU2.jpg)

For more please refer to [1] and I strongly suggest to look [2] as well to see how SPP layer behaves.

[1] Mendonça, S. (2015). Splitting, parallel gradient and Bakry-Emery Ricci curvature. Differential Geometry. Retrieved from http://arxiv.org/abs/1502.0185

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2014). Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, 1–14. Computer Vision and Pattern Recognition. Retrieved from http://arxiv.org/abs/1406.4729


### Related posts:

1. [Brief History of Machine Learning](http://www.erogol.com/brief-history-machine-learning/ "Brief History of Machine Learning")
3. [Recent Advances in Deep Learning](http://www.erogol.com/recent-advances-in-deep-learning/ "Recent Advances in Deep Learning")
4. [Recent Advances in Deep Learning #2](http://www.erogol.com/recent-advances-in-deep-learning-2/ "Recent Advances in Deep Learning #2")