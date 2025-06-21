import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import kornia.augmentation as K
import kornia
from easyio import show, load, save
import glob
import random

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_file, partition_file, full_class_label_file_glob, use_all_classes=False, partition=None, transform=None):
        self.image_dir = image_dir
        self.use_all_classes = use_all_classes
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        
        df_labels = pd.read_csv(label_file, sep=" ", names=["file", "label"])
        df_labels.index = df_labels["file"].apply(lambda x: x.split(".")[0])
        
        df_partitions = pd.read_csv(partition_file, sep=" ", names=["file", "partition"])
        df_partitions.index = df_partitions["file"].apply(lambda x: x.split(".")[0])
        df_partitions = df_partitions.drop(columns=["file"])
        
        df_attacks = pd.concat(pd.read_csv(p, sep=" ")for p in Path().glob(full_class_label_file_glob))
        df_attacks.index = df_attacks["attacked_image"].apply(lambda x: x.split(".")[0])
        df_attacks = df_attacks[["attack_id"]]
        
        self.data = df_labels.join(df_partitions).join(df_attacks)
        self.data["attack_id"] = self.data["attack_id"].fillna(-1).astype(int) + 1
        if partition:
            self.data = self.data[self.data.partition == partition]
        self.data = self.data.to_dict("records")
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data[idx]["file"])
        if img_path not in self.cache:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.cache[img_path] = image
        else:
            image = self.cache[img_path]
        if self.use_all_classes:
            return image, self.data[idx]["attack_id"]
        else:
            return image, self.data[idx]["label"]

class CustomPairDataset(Dataset):
    def __init__(self, image_dir, transform=None, train=True, split_ratio=0.8, seed=42):
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.train = train
        self.cache = {}

        # Find all pair directories
        all_pairs = sorted(glob.glob(os.path.join(image_dir, "pair_*_label_*")))

        # Optional train/test split
        random.seed(seed)
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * split_ratio)
        if train:
            self.pairs = all_pairs[:split_idx]
        else:
            self.pairs = all_pairs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_dir = self.pairs[idx]
        label = int(pair_dir.split("_")[-1])  # Extract y from the directory name

        # Load adversarial and benign images
        im0_path = glob.glob(os.path.join(pair_dir, "im_0*"))[0]
        im1_path = glob.glob(os.path.join(pair_dir, "im_1*"))[0]

        if im0_path not in self.cache:
            image0 = Image.open(im0_path).convert("RGB")
            self.cache[im0_path] = self.transform(image0)
        image0 = self.cache[im0_path]

        if im1_path not in self.cache:
            image1 = Image.open(im1_path).convert("RGB")
            self.cache[im1_path] = self.transform(image1)
        image1 = self.cache[im1_path]

        # Return both images and the label
        return image0, image1, label  