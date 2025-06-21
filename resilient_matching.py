import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn import metrics
from facenet_pytorch import InceptionResnetV1
from kornia_transformations import KorniaTransformations

def verify(embedding0, embedding1, p=2, thres=1.0):
    distance = (embedding0 - embedding1).norm(p=p).item()
    if distance < thres:
        return 1
    else:
        return 0
    
image_dir = "data/AdvLFW/images"
split_ratio = 1.0
seed = 42
    
train_dataset = CustomPairDataset(
    image_dir, 
    train = True, 
    split_ratio = split_ratio, 
    seed = seed,
)

transform = transforms.Compose([transforms.ToTensor()])
device = "cuda"

resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

correct = 0
wrong = 0
with torch.no_grad():
    for im0, im1, label in tqdm(train_dataset):
        im0 = KorniaTransformations.gaussian_blur(im0.unsqueeze(0), default=True).cuda()
        im1 = KorniaTransformations.gaussian_blur(im1.unsqueeze(0), default=True).cuda()
        embedding0 = resnet(im0)
        embedding1 = resnet(im1)
        res = verify(embedding0, embedding1, thres=1.0)
        if label == res:
            correct +=1
        else:
            wrong +=1
        
print("correct: " + str(correct))        
print("wrong: " + str(wrong))
