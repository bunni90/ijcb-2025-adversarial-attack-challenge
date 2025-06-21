import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from rich.progress import Progress
import rich
import torchvision
import kornia.augmentation as K
import kornia
from sklearn.utils.class_weight import compute_class_weight
from easyio import show, load, save
from sklearn import metrics
import matplotlib.pyplot as plt
from models import Resnet18MoreThanRGB, ModelMultiToBinaryWrapper

from dataloading import CustomImageDataset

image_dir = "data/AdvCelebA/images"
label_file = "data/AdvCelebA/attack_CelebA.txt"
partition_file = "data/AdvCelebA/list_eval_partition_no_overlap.txt"
full_class_label_file_glob = "data/AdvCelebA/final_attack_attackid_*.txt"
# num_classes = 2
num_classes = 11

train_dataset = CustomImageDataset(
    image_dir, 
    label_file, 
    partition_file, 
    full_class_label_file_glob,
    use_all_classes=True,
    partition=0,
)

test_dataset = CustomImageDataset(
    image_dir, 
    label_file, 
    partition_file, 
    full_class_label_file_glob,
    use_all_classes=True,
    partition=2,
)
            
epochs = 20
lr = 0.0001
batch_size = 16

model = Resnet18MoreThanRGB(num_classes).cuda(); model.image_size=112

model = ModelMultiToBinaryWrapper(model, num_classes).cuda()

class_weights = torch.tensor([0.5]+[1.5]*10)

criterion_multi = nn.CrossEntropyLoss(weight=class_weights).cuda()
criterion_binary = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.01])).cuda()
optimizer = optim.AdamW(model.parameters(), lr=lr)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

aug = K.AugmentationSequential(K.auto.RandAugment(n=2, m=15))

model.train()

with Progress() as progress:
    for epoch in range(epochs):
        task = progress.add_task(None, total=len(train_dataloader))
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.cuda(), labels.cuda()
            images = aug(images)
            images = torchvision.transforms.functional.resize(images, (model.image_size, model.image_size))

            optimizer.zero_grad()
            outputs_multi, outputs_binary = model(images, return_binary_logits=True)
            loss = criterion_multi(outputs_multi, labels)
            loss += criterion_binary(outputs_binary, labels.minimum(torch.ones_like(labels)))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs_multi.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress.update(task, advance=1, description=f"[bold cyan]Epoch {epoch+1}[/bold cyan] - Loss: {total_loss / total:.4f} | Acc: {correct / total:.2%}")
