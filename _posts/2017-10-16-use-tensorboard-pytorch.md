---
layout: post
title: "How to use Tensorboard with PyTorch"
description: "Let's directly dive in"
tags: embedding plotting pytorch tensorboard
minute: 2
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Let's directly dive in. The thing here is to use Tensorboard to plot your PyTorch trainings. For this, I use [TensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX) which is a nice interface communicating Tensorboard avoiding Tensorflow dependencies.

First install the requirements;

```python

pip install tensorboard
pip install tensorboardX
```

Things thereafter very easy as well, but you need to know how you need to communicate with the board to show your training and it is not that easy, if you don't know Tensorboard hitherto.

```python

...
from tensorboardX import SummaryWriter
...

writer = SummaryWriter('your/path/to/log_files/') 

...
# in training loop
writer.add_scalar('Train/Loss', loss, num_iteration)
writer.add_scalar('Train/Prec@1', top1, num_iteration) 
writer.add_scalar('Train/Prec@5', top5, num_iteration) 

...
# in validation loop
writer.add_scalar('Val/Loss', loss, epoch) 
writer.add_scalar('Val/Prec@1', top1, epoch)
writer.add_scalar('Val/Pred@5', top5, epoch)  
```

You can also see the embedding of your dataset

```python

from torchvision import datasets
from tensorboardX import SummaryWriter

dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
```

This is also how you can plot your model graph. The important part is to give the output tensor to writer as well with you model. So that, it computes the tensor shapes in between. I also need to say, it is very slow for large models.

```python

import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)+F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

model = Mnist()

# if you want to show the input tensor, set requires_grad=True
res = model(torch.autograd.Variable(torch.Tensor(1,1,28,28), requires_grad=True))

writer = SummaryWriter()
writer.add_graph(model, res)

writer.close()
```

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Duplicate Question Detection with Deep Learning on Quora Dataset](http://www.erogol.com/duplicate-question-detection-deep-learning/ "Duplicate Question Detection with Deep Learning on Quora Dataset")
2. [SPP network for Pytorch](http://www.erogol.com/spp-network-pytorch/ "SPP network for Pytorch")