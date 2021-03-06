#!/usr/bin/python
# encoding: utf-8

import torch
from torch.utils.data import Dataset
from utils import read_truths_args
from image import *


class YoloDataset(Dataset):
    """Yolo dataset format.
    For training and testing, the format of data needs to be unified.
    Arguments:
        root(str): txt file includes all image paths
        shape(tuple): YOLOv3 needs fixed size of image, official size is (416, 416)
        jitter(float): for data augmentation
        hue(float): for data augmentation
        saturation(float): for data augmentation
        exposure(float): for data augmentation
        transform(pytorch transformation):  image transformation, ToTensor() always included
        train(bool): train or test

    """
    def __init__(self, root, shape=None, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5, transform=None, train=True):

        with open(root, 'r') as file:   # 图片信息
            self.lines = file.readlines()

        self.nSamples = len(self.lines) # 图片个数
        self.transform = transform
        self.train = train
        self.shape = shape

        self.jitter = jitter
        self.hue = hue                      # 控制色调
        self.saturation = saturation        # 控制饱和度
        self.exposure = exposure            # 控制曝光度

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        if self.train:
            img, label = load_data_detection(imgpath, self.shape,  self.jitter, self.hue, self.saturation,
                                             self.exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img, org_w, org_h = letter_image(img, self.shape[0], self.shape[1]), img.width, img.height
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels')\
                .replace('.jpg', '.txt').replace('.png', '.txt')
            # one image at most has 50 bounding boxes, we need to have a fixed size
            label = torch.zeros(50 * 5)
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0 / img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1, 5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()           # element number
            if tsz > 50 * 5:
                label = tmp[0:50 * 5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, label
        else:
            # we need to transfer image to original size to evaluate perfoemance
            # so neet to record original size
            return img, label, org_w, org_h
