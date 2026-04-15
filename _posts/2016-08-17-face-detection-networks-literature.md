---
layout: post
title: "Face Detection by Literature"
description: "Please ping me if you know something more"
tags: deep learning face detection
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

*Please ping me if you know something more.*

**Multi-view Face Detection Using Deep Convolutional Neural Network**

1. Train face classifier with face (> 0.5 overlap) and background (<0.5 overlap) images.
2. Compute heatmap over test image scaled to different sizes with sliding window
3. Apply NMS .
4. Computation intensive, especially for CPU.

* http://arxiv.org/abs/1502.02766

![multiview_face](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/multiview_face.png)

**From Facial Parts Responses to Face Detection: A Deep Learning Approach**

Keywords: object proposals, facial parts,  more annotation.

1. Use facial part annotations
2. Bottom up to detect face from facial parts.
3. "Faceness-Net’s pipeline consists of three stages,i.e. generating partness maps, ranking candidate windows by faceness scores, and refining face proposals for face detection."
4. Train part based classifiers based on attributes related to different parts of the face i.e. for hair part train ImageNet pre-trained network for color classification.
5. Very robust to occlusion and background clutter.
6. To much annotation effort.
7. Still object proposals (DL community should skip proposal approach. It complicate the problem by creating a new domain of problem :)) ).

* http://arxiv.org/abs/1509.06451

![facial_parts](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/facial_parts.png)

**Supervised Transformer Network for Efficient Face Detection**

* http://home.ustc.edu.cn/~chendong/STN\_Detector/stn\_detector.pdf

**UnitBox: An Advanced Object Detection Network**

* http://arxiv.org/abs/1608.02236

**Deep Convolutional Network Cascade for Facial Point Detection**

* http://www.cv-foundation.org/openaccess/content\_cvpr\_2013/papers/Sun\_Deep\_Convolutional\_Network\_2013\_CVPR\_paper.pdf
* http://mmlab.ie.cuhk.edu.hk/archive/CNN\_FacePoint.htm
* https://github.com/luoyetx/deep-landmark

**WIDER FACE: A Face Detection Benchmark**

A novel cascade detection method being a state of art at WIDER FACE

1. Train separate CNNs for small range of scales.
2. Each detector has two stages; Region Proposal Network + Detection Network

* http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
* http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/paper.pdf

![face_wider](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2016/08/face_wider.png)

**DenseBox (DenseBox: Unifying Landmark Localization with End to End Object Detection)**

Keywords: upsampling, hardmining, no object proposal, BAIDU

1. Similar to YOLO .
2. Image pyramid of input
3. Feed to network
4. Upsample feature maps after a layer.
5. Predict classification score and bbox location per pixel on upsampled feature map.
6. NMS to bbox locations.
7. SoA at MALF face dataset

* http://arxiv.org/pdf/1509.04874v3.pdf
* http://www.cbsr.ia.ac.cn/faceevaluation/results.html

**Face Detection without Bells and Whistles**

Keywords: no NN, DPM, Channel Features

1. ECCV 2014
2. Very high quality detections
3. Very slow on CPU and acceptable on GPU

* https://bitbucket.org/rodrigob/doppia/
* http://rodrigob.github.io/documents/2014\_eccv\_face\_detection\_with\_supplementary\_material.pdf


### Related posts:

1. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
2. [Methods used by us as Qualcomm Research at ImageNet 2015](http://www.erogol.com/methods-used-us-qualcomm-research-imagenet-2015/ "Methods used by us as Qualcomm Research at ImageNet 2015")
3. [Paper review: Dynamic Capacity Networks](http://www.erogol.com/1314-2/ "Paper review: Dynamic Capacity Networks")
4. [Why do we need better word representations ?](http://www.erogol.com/need-better-word-representations/ "Why do we need better word representations ?")