---
layout: post
title: "Object Detection Literature"
description: "<Please let me know if there are more works comparable to these below"
tags: computer vision deep learning detection
minute: 3
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

*<Please let me know if there are more works comparable to these below.>*

**R-CNN minus R**

* http://arxiv.org/pdf/1506.06981.pdf

**FasterRCNN (Faster R-CNN: Towards Real-Time Object**  
**Detection with Region Proposal Networks)**

Keywords: RCNN, RoI pooling, object proposals, ImageNet 2015 winner.

PASCAL VOC2007: 73.2%

PASCAL VOC2012: 70.4%

ImageNet Val2 set: 45.4% MAP

1. Model agnostic
2. State of art with Residual Networks
   * http://arxiv.org/pdf/1512.03385v1.pdf
3. Fast enough for oflline systems and partially for inline systems

* https://arxiv.org/pdf/1506.01497.pdf
* https://github.com/ShaoqingRen/faster\_rcnn (official)
* https://github.com/rbgirshick/py-faster-rcnn
* http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf
* https://github.com/precedenceguo/mx-rcnn
* https://github.com/mitmul/chainer-faster-rcnn
* https://github.com/andreaskoepf/faster-rcnn.torch

**YOLO (You Only Look Once: Unified, Real-Time Object Detection)**

Keywords: real-time detection, end2end training.

PASCAL VOC 2007: 63,4% (YOLO), 57.9% (Fast YOLO)

RUN-TIME : 45 FPS (YOLO), 155 FPS (Fast YOLO)

1. VGG-16 based model
2. End-to-end learning with no extra hassle (no proposals)
3. Fastest with some performance payback relative to Faster RCNN
4. Applicable to online systems

* http://pjreddie.com/darknet/yolo/
* https://github.com/pjreddie/darknet
* https://github.com/BriSkyHekun/py-darknet-yolo (python interface to darknet)
* https://github.com/tommy-qichang/yolo.torch
* https://github.com/gliese581gg/YOLO\_tensorflow
* https://github.com/ZhouYzzz/YOLO-mxnet
* https://github.com/xingwangsfu/caffe-yolo
* https://github.com/frankzhangrui/Darknet-Yolo (custom training)

**MultiBox (Scalable Object Detection using Deep Neural Networks)**

Keywords: cascade classifiers, object proposal network.

1. Similar to YOLO
2. Two successive networks for generating object proposals and classifying these

* http://www.cv-foundation.org/openaccess/content\_cvpr\_2014/papers/Erhan\_Scalable\_Object\_Detection\_2014\_CVPR\_paper.pdf
* https://github.com/google/multibox
* https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html

**ION (Inside - Outside Net)**

Keywords: object proposal network, RNN, context features

1. RNN networks on top of conv5 layer in 4 different directions
2. Concate different layer features with L2 norm + rescaling

* (great slide) http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf

**UnitBox ( UnitBox: An Advanced Object Detection Network)**

* https://arxiv.org/pdf/1608.01471v1.pdf

**DenseBox (DenseBox: Unifying Landmark Localization with End to End Object Detection)**

Keywords: upsampling, hardmining, no object proposal, BAIDU

1. Similar to YOLO .
2. Image pyramid of input
3. Feed to network
4. Upsample feature maps after a layer.
5. Predict classification score and bbox location per pixel on upsampled feature map.
6. NMS to bbox locations.

* http://arxiv.org/pdf/1509.04874v3.pdf

**MRCNN: Object detection via a multi-region & semantic segmentation-aware CNN model**

PASCAL VOC2007: 78.2% MAP

PASCAL VOC2012: 73.9% MAP

Keywords: bbox regression, segmentation aware

1. very large model and so much detail.
2. Divide each detection windows to different regions.
3. Learn different networks per region scheme.
4. Empower representation by using the entire image network.
5. Use segmentation aware network which takes the etnrie image as input.

* http://arxiv.org/pdf/1505.01749v3.pdf
* https://github.com/gidariss/mrcnn-object-detection

**SSD: Single Shot MultiBox Detector**

PASCAL VOC2007: 75.5% MAP (SSD 500), 72.1% MAP (SSD 300)

PASCAL VOC2012: 73.1% MAP (SSD 500)

RUN-TIME: 23 FPS (SSD 500), 58 FPS (SSD 300)

Keywords: real-time, no object proposal, end2end training

1. Faster and accurate then YOLO (their claim)
2. Not useful for small objects

* https://arxiv.org/pdf/1512.02325v2.pdf
* https://github.com/weiliu89/caffe/tree/ssd

![Results for SSD, YOLO and F-RCNN](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/detection_results.png)

Results for SSD, YOLO and F-RCNN

**CRAFT (**CRAFT Objects from Images)****

PASCAL VOC2007: 75.7% MAP

PASCAL VOC2012: 71.3% MAP

ImageNet Val2 set: 48.5% MAP

* intro: CVPR 2016. Cascade Region-proposal-network And FasT-rcnn. an extension of Faster R-CNN
* http://byangderek.github.io/projects/craft.html
* https://github.com/byangderek/CRAFT
* https://arxiv.org/abs/1604.03239


### Related posts:

1. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
2. [How does Feature Extraction work on Images?](http://www.erogol.com/feature-extraction-work-images/ "How does Feature Extraction work on Images?")
3. [Harnessing Deep Neural Networks with Logic Rules](http://www.erogol.com/harnessing-deep-neural-networks-with-logic-rules/ "Harnessing Deep Neural Networks with Logic Rules")
4. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")