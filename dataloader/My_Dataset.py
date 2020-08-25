# coding: utf-8
import torch
import os
import numpy.random as nr
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import csv

from torchvision import transforms
from torchvision.transforms import functional

class My_Dataset(object):
    def __init__(self, csv_path, phase, num_classes, padding=False):

        self.csv_path = csv_path
        self.phase = phase
        self.num_classes = num_classes
        self.padding = padding
        self.scale = []

        self.images, self.labels = self.prepare_data()

        if self.phase == 'train':
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
            ])
        elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
            self.trans = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise IndexError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)

        if self.padding:  # 调整图像长边为224，以下代码出自torchvision.transforms.functional.resize
            size = 224
            w, h = image.size
            if max(w, h) == size:
                ow, oh = w, h
                pass
            elif w < h:
                ow = int(size * w / h)
                oh = size
                image = image.resize((ow, oh), resample=Image.BILINEAR)
            else:
                ow = size
                oh = int(size * h / w)
                image = image.resize((ow, oh), resample=Image.BILINEAR)

            # 将短边补齐到224
            image = functional.pad(image, fill=0,
                                   padding=((size - ow) // 2, (size - oh) // 2,
                                            (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))
        else:  # resize到224*224
            image = functional.resize(image, (224, 224))

        image = self.trans(image)

        label = self.labels[index]

        return image, label, image_path

    def prepare_data(self):
        lines = []
        with open(self.csv_path, 'r') as f:
            lines.extend(f.readlines())
        f.close()

        if self.phase == 'train':
            random.shuffle(lines)
        elif self.phase == 'val' or self.phase == 'test':
            pass
        else:
            raise ValueError

        images = [str(x).strip().split(',')[0] for x in tqdm(lines, desc='Preparing Images')]
        labels = [int(str(x).strip().split(',')[1]) for x in lines]

        return images, labels

    def dist(self):
        dist = {}
        for l in tqdm(self.labels, desc="Counting data distribution"):
            if str(l) in dist.keys():
                dist[str(l)] += 1
            else:
                dist[str(l)] = 1
        return dist
