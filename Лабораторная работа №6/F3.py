import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter



resnet18 = models.resnet18(False)
writer = SummaryWriter('pa')
     
    

dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[100:200].float()
label = dataset.test_labels[100:200]

features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
writer.close()
