import torch
import torch.nn as nn
import torch.optim as optim
import crack_dataset as DS
import copy
import os
import time
import logging
from tools.focal_loss import focal_binary_cross_entropy
from torch.nn import BCEWithLogitsLoss



def get_zennet(arch_path, num_classes, use_SE):
    """
    load the Zen-NAS searched model from stored arch planetext

    :param arch_path: path for model architecture description file.txt
    :param num_classes: the data class number
    :param use_SE: whether to use Squeeze-and-Excitation module
    """
    from ref_codes.ZenNAS.ZenNet import masternet
    
    with open(arch_path, 'r') as fid:
        model_plainnet_str = fid.readline().strip()
    
    model = masternet.PlainNet(num_classes=num_classes, plainnet_struct=model_plainnet_str, use_se=use_SE)
    return model

def get_ZenNet_pretrained(model_name, num_classes):
    from ref_codes.ZenNAS.ZenNet import get_ZenNet
    from ref_codes.ZenNAS.PlainNet import basic_blocks
    model = get_ZenNet(model_name, pretrained=True)
    
    # adjust the last layer to adapt to the new class number
    model.fc_linear = basic_blocks.Linear(in_channels=model.fc_linear.in_channels, out_channels=num_classes)
    return model

def show_label(label):
    from tabulate import tabulate
    label = label[0]
    d = [ ["Background", label[0]],
     ["Crack", label[1]],
     ["Spallation", label[2]],
     ["Efflorescence", label[3]],
     ["ExposedBars", label[4]],
     ["CorrosionStain", label[5]]]
    print(tabulate(d, headers=["Sort", "0/1"]))

def show_sample(data):
    import torch
    from torchvision.transforms import ToPILImage
    show = ToPILImage() 
    show(data[0]).show()

def get_sample():
    """
    load a random sample for test

    :param path: path for the test picture
    
    return data, label
    """
    import datasets as DS
    class Args:
      # dataset_path = "/storage/data/classification_dataset_balanced/"
      dataset_path = "./sample/"
      patch_size = 1
      batch_size = 1
      workers = 1
      def __init__(self, patch_size, batch_size, workers):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.workers = workers
    args = Args(224, 1, 1)
    dataset = DS.CODEBRIM(torch.cuda.is_available(),args)
    dataLoaders = {'sample': dataset.train_loader}
    sample = iter(dataLoaders['sample']).next()
    return sample

