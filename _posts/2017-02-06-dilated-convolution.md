---
layout: post
title: "Dilated Convolution"
description: "In simple terms, dilated convolution is just a convolution applied to input with defined gaps"
tags: convolution deep learning dilated convolution dilation note
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

In simple terms, dilated convolution is just a convolution applied to input with defined gaps. With this definitions, given our input is an 2D image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels. The best to see the figures below with the same k values.

The figure below shows dilated convolution on 2D data. Red dots are the inputs to a filter which is 3x3 in this example, and greed area is the receptive field captured by each of these inputs. Receptive field is the implicit area captured on the initial input by each input (unit) to the next layer .

![](http://www.inference.vc/content/images/2016/05/Screen-Shot-2016-05-12-at-09-47-12.png)

Dilated convolution is a way of increasing receptive view (global view) of the network exponentially and linear parameter accretion. With this purpose, it finds usage in applications cares more about integrating knowledge of the wider context with less cost.

One general use is image segmentation where each pixel is labelled by its corresponding class. In this case, the network output needs to be in the same size of the input image. Straight forward way to do is to apply convolution then add deconvolution layers to upsample[1]. However, it introduces many more parameters to learn. Instead, dilated convolution is applied to keep the output resolutions high and it avoids the need of upsampling [2][3].

Dilated convolution is applied in domains beside vision as well. One good example is WaveNet[4] text-to-speech solution and ByteNet learn time text translation. They both use dilated convolution in order to capture global view of the input with less parameters.

![](http://dlacombejr.github.io/assets/CAX_blog/dilated_convolution.jpg)

From [5]

In short, dilated convolution is a simple but effective idea and you might consider it in two cases;

1. Detection of fine-details by processing inputs in higher resolutions.
2. Broader view of the input to capture more contextual information.
3. Faster run-time with less parameters

[1] Long, J., Shelhamer, E., & Darrell, T. (2014). Fully Convolutional Networks for Semantic Segmentation. Retrieved from http://arxiv.org/abs/1411.4038v1

[2]Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2014). Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs. *Iclr*, 1–14. Retrieved from http://arxiv.org/abs/1412.7062

[3]Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *Iclr*, 1–9. http://doi.org/10.16373/j.cnki.ahr.150049

[4]Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., … Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio, 1–15. Retrieved from http://arxiv.org/abs/1609.03499

[5]Kalchbrenner, N., Espeholt, L., Simonyan, K., Oord, A. van den, Graves, A., & Kavukcuoglu, K. (2016). Neural Machine Translation in Linear Time. *Arxiv*, 1–11. Retrieved from http://arxiv.org/abs/1610.10099

[Share](https://www.addtoany.com/share)

### Related posts:

1. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
2. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
3. [Paper review: Dynamic Capacity Networks](http://www.erogol.com/1314-2/ "Paper review: Dynamic Capacity Networks")
4. [Selfai: A Method for Understanding Beauty in Selfies](http://www.erogol.com/selfai-predicting-facial-beauty-selfies/ "Selfai: A Method for Understanding Beauty in Selfies")