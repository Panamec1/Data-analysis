import torchvision.models as models
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from tensorboardX import SummaryWriter
alexnet = models.alexnet(pretrained=True)

from torchvision import transforms
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                    )])
img = Image.open("dog.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

alexnet.eval()
out = alexnet(batch_t)

with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]
  
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())

im=[]
l=[]

for filename in glob.glob(os.path.join("C://Users//User\Desktop\Предметы\Д.з\Курс III\Анализ данных\Лабораторная работа №6\data", '*.jpg')):
    img = Image.open(filename)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    alexnet.eval()
    out = alexnet(batch_t)
    _, index = torch.max(out, 1)
    im.append(img)
    l.append(labels[index[0]])


writer = SummaryWriter('i')
writer.add_embedding(len(l), metadata=l, label_img=im)
