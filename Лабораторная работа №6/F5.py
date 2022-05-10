import random
import numpy as np

import torch
import torch.nn as nn

import scipy

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, SBDataset, SVHN
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tensorboardX import SummaryWriter

from tqdm.notebook import tqdm, trange

random.seed(239)
np.random.seed(239)
torch.manual_seed(239)
torch.cuda.manual_seed(239)
torch.backends.cudnn.deterministic = True


OUTPUT_DIM = 10
EPOCH_NUM = 4  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


basic_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])


train_dataset = SVHN(
    root='./SVHN/train',
    split='train',
    transform=basic_transform,
    download=True
)

test_dataset = SVHN(
    root='./SVHN/test',
    split='test',
    transform=transforms.ToTensor(),  
    download=True
)


dataloaders = {
    'train': DataLoader(
        dataset=train_dataset,
        batch_size=128, 
        shuffle=True
    ),
    'val': DataLoader(
        dataset=test_dataset,
        batch_size=128, 
        shuffle=False
    )
}



class SVHNClassifier(nn.Module):
    def __init__(self, ouput_dim):
        super(SVHNClassifier, self).__init__()  
        self.model = resnet50(pretrained=True)  
        
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64, ouput_dim)
        )
    
    def embed(self, x):
        return self.fc1(self.model(x))
    
    def forward(self, x):
        resnet_out = self.embed(x)
        return self.fc2(resnet_out)


model = SVHNClassifier(OUTPUT_DIM)
model = model.to(DEVICE)



train_items = [train_dataset[i] for i in range(1000)]
xs, ys = zip(*train_items)

model = model.eval()
xs = model (xs)
xs = torch.stack(xs)
xs = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))(xs)


writer = SummaryWriter('o')
features = xs.mean(dim=1).view(-1, 32 * 32)
writer.add_embedding(features,
                     metadata=ys,
                     label_img=xs)
writer.close()
