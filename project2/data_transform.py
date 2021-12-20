# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:06:55 2021

@author: syson
"""

import torchvision.transforms as transforms
import datasets as DS
import random
import torch
#geometric transform
#0->RandomHorizontalFlip
#1->RandomVerticalFlip
#2->RandomRotation
#3->RandomResizedCrop

geometric_transform_list = []
RandomHorizontalFlip = transforms.Compose([
    transforms.Resize(224),  # mandate
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor()])  # mandate
geometric_transform_list.append(RandomHorizontalFlip)

RandomVerticalFlip = transforms.Compose([
    transforms.Resize(224),  # mandate
    transforms.RandomVerticalFlip(p = 1),
    transforms.ToTensor()])  # mandate
geometric_transform_list.append(RandomVerticalFlip)

RandomRotation = transforms.Compose([
    transforms.Resize(224),  # mandate
    transforms.RandomRotation(360),
    transforms.ToTensor()])  # mandate
geometric_transform_list.append(RandomRotation)

RandomResizedCrop = transforms.Compose([
    transforms.Resize(256),  # mandate
    transforms.RandomResizedCrop(size = 224),
    transforms.ToTensor()])  # mandate
geometric_transform_list.append(RandomResizedCrop)

#photometric transform
#0->ColorJitter
#1->GaussianBlur
#2->RandomAdjustSharpness
#3->Normalize
photometric_transform_list = []
ColorJitter = transforms.Compose([
    transforms.Resize(224),  # mandate
    transforms.ColorJitter(brightness=random.uniform(0, 1), 
                           contrast=random.uniform(0, 1), 
                           saturation=random.uniform(0, 1), 
                           hue=0),
    transforms.ToTensor()])  # mandate
photometric_transform_list.append(ColorJitter)

GaussianBlur = transforms.Compose([#?torch.nn.Sequential
    transforms.Resize(224),  # mandate
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor()])  # mandate
photometric_transform_list.append(GaussianBlur)

RandomAdjustSharpness = transforms.Compose([#?torch.nn.Sequential
    transforms.Resize(224),  # mandate
    transforms.RandomAdjustSharpness(sharpness_factor = 2, p=1),
    transforms.ToTensor()])  # mandate
photometric_transform_list.append(RandomAdjustSharpness)

Normalize = transforms.Compose([#?torch.nn.Sequential
    transforms.Resize(224),  # mandate
    transforms.Normalize((0.499, 0.559, 0.535), (0.021, 0.018, 0.019)),
    transforms.ToTensor()])  # mandate
photometric_transform_list.append(Normalize)

#mode
#-1->randomly select a transform
#0->RandomHorizontalFlip
#1->RandomVerticalFlip
#2->RandomRotation
#3->RandomResizedCrop
#4->ColorJitter
#5->GaussianBlur
#6->RandomAdjustSharpness
#7->Normalize
def random_transform(input_batch, mode):
    for img in enumerate(input_batch):
        choice = mode
        if choice == -1:#mix mode
            choice = random.randint(0, 8)          
        if(choice < 4):
            img = geometric_transform_list[choice](img)
        if(choice >=4 and choice < 8):
            img = photometric_transform_list[choice-4](img)
    return input_batch
def getDataSet(patch_size, batch_size, workers=8):
    # build a class to satisfy the input of loader producer provided by the paper
    class Args:
      dataset_path = "/storage/data/classification_dataset_balanced/"
      patch_size = 1
      batch_size = 1
      workers = 1
      def __init__(self, patch_size, batch_size, workers):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.workers = workers
    args = Args(patch_size, batch_size, workers)
    # use the loader producer from the paper
    dataset = DS.CODEBRIM(torch.cuda.is_available(),args)
    dataLoaders = {'train': dataset.train_loader, 'val': dataset.val_loader, 'test':dataset.test_loader}
    return dataLoaders
dataLoader = getDataSet(224, 16)


