---
layout: post
title: "Installing OpenCV 3.2 to Anaconda Environment with ffmpeg Support"
description: "Sometimes, It is really a mess to try installing OpenCV to your system"
tags: code codebook installation opencv
minute: 1
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Sometimes, It is really a mess to try installing OpenCV to your system. Nevertheless, it is really great library for any case of vision and you are obliged to use it. (No complain, just C++).

I try to list my commands here in a sequence  and hope it will work for you too.

#### Install dependencies

```python


apt install gcc g++ git libjpeg-dev libpng-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev pkg-config cmake libgtk2.0-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavcodec-dev libavutil-dev libavfilter-dev libavformat-dev libavresample-dev

conda install libgcc

```

#### Download OpenCV

```python


//First, go to your folder to hosting installation
wget https://github.com/Itseez/opencv/archive/3.2.0.zip

unzip 3.2.0.zip
cd opencv-3.2.0

mkdir build
cd build

```

##### Cmake and Setup Opencv

This cmake command targets python3.x and your target virtual environment. Therefore, before running it activate your environment. Do not forget to check flags depending on your case.

```python


cmake -DWITH_CUDA=OFF -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DENABLE_AVX=ON -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_IPP=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_VTK=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_opencv_python2=OFF -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -DPYTHON3_EXECUTABLE=$(which python3) -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D PYTHON_EXECUTABLE=~/miniconda3/envs/dl/bin/python -D BUILD_EXAMPLES=ON ..

make -j 4

sudo make install

```

Then check your installation on Python

```python


import cv2

print(cv2.__version__) # should output opencv-3.2.0

```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Foreground Extraction and Contour Detection with OpenCV 3](http://www.erogol.com/foreground-extraction-and-contour-detection-with-opencv-3/ "Foreground Extraction and Contour Detection with OpenCV 3")
2. [Timeout function if it takes too long to finish in Python](http://www.erogol.com/timeout-function-takes-long-finish-python/ "Timeout function if it takes too long to finish in Python")
3. [Pull all repository with all submodules](http://www.erogol.com/pull-all-repository-with-all-submodules/ "Pull all repository with all submodules")
4. [How to use Python Decorators](http://www.erogol.com/use-python-decorators/ "How to use Python Decorators")