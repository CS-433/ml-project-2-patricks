########################
# importing libraries
########################
# system libraries
import numpy as np
import os
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import xml.etree.ElementTree as ElementTree


class CODEBRIMSplit(datasets.ImageFolder):
    """
    definition of class for reading data-split images and class labels, and iterating
    over the datapoints

    Parameters:
        root (string): directory path for the data split
        xml_list (list): list of paths to xmls for defect and background meta-data
        transform (torchvision.transforms.Compose): transforms for the input data
        target_transform (callable): transform for the targets
        loader (callable): for loading an image given its path

    Attributes:
        file_list (dictionary): dictionary of file names (keys) and the corresponding
                                class labels (values)
        num_classes (int): number of classes in the dataset (6)
    """
    def __init__(self, root, xml_list, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(CODEBRIMSplit, self).__init__(root, transform, target_transform, loader)
        self.file_list = {}
        self.num_classes = 6
        for file_name in xml_list:
            last_dot_idx = file_name.rfind('.')
            f_name_idx = file_name.rfind('/')
            root_path = file_name[f_name_idx + 1: last_dot_idx]
            tree = ElementTree.parse(file_name)
            root = tree.getroot()
            for defect in root:
                crop_name = list(defect.attrib.values())[0]
                target = self.compute_target_multi_target(defect)
                self.file_list[os.path.join(root_path, crop_name)] = target

    def __getitem__(self, idx):
        """
        defines the iterator for the dataset and returns datapoints in the form of tuples

        Parameters:
            idx (int): index to return the datapoint from

        Returns:
            a datapoint tuple (sample, target) for the index
        """
        image_batch = super(CODEBRIMSplit, self).__getitem__(idx)[0]
        image_name = self.imgs[idx][0]
        f_name_idx = image_name.rfind('/')
        f_dir_idx = image_name[: f_name_idx].rfind('/')
        de_lim = image_name.rfind('_-_')
        file_type = image_name.rfind('.')
        if de_lim != -1:
            name = image_name[f_dir_idx + 1: de_lim] + image_name[file_type:]
        else:
            name = image_name[f_dir_idx + 1:]
        return [image_batch, self.file_list[name]]

    def compute_target_multi_target(self, defect):
        """
        enumerates the class-label by defining a float32 numpy array

        Parameters:
            defect (string): the class labels in the form of a string

        Returns:
            the enumerated version of the labels in the form of a numpy array 
        """
        out = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            if defect[i].text == '1':
                out[i] = 1.0
        return out


class CODEBRIM:
    """
    definition of CODEBRIM dataset, train/val/test splits, train/val/test loaders

    Parameters:
        args (argparse.Namespace): parsed command line arguments
        is_gpu (bool): if computational device is gpu or cpu

    Attributes:
        num_classes (int): number of classes in the dataset (= 6)
        dataset_path (string): path to dataset folder
        dataset_xml_list (list): list to dataset meta-data
        train_set (CODEBRIMSplit): train split
        val_set (CODEBRIMSplit): validation split
        test_set (CODEBRIMSplit): test split
        train_loader (torch.utils.data.DataLoader): data-loader for train-split
        val_loader (torch.utils.data.DataLoader): data-loader for val-split
        test_loader (torch.utils.data.DataLoader): data-loader for test-split
    """
    def __init__(self, is_gpu, args):
        self.num_classes = 6
        self.dataset_path = args.dataset_path
        self.mode = args.mode
        self.dataset_xml_list = [os.path.join(args.dataset_path, 'metadata/background.xml'),
                                 os.path.join(args.dataset_path, 'metadata/defects.xml')]
        self.train_set, self.val_set, self.test_set = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset_loader(args.batch_size, args.workers,
                                                                                       is_gpu)

    def get_dataset(self, patch_size):
        """
        return dataset splits

        Parameters:
            patch_size (int): patch-size to rescale the images to
        
        Returns:
            train_set, val_set, test_set of type lib.Datasets.datasets.CODEBRIMSplit
        """
        #geometric transform
        #0->RandomHorizontalFlip
        #1->RandomVerticalFlip
        #2->RandomRotation
        #3->RandomResizedCrop
        #4->RandomPerspective
        transform_list = []
		#0->RandomHorizontalFlip
        RandomHorizontalFlip = transforms.Compose([
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(p = 1),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomHorizontalFlip)
		#1->RandomVerticalFlip
        RandomVerticalFlip = transforms.Compose([
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.RandomVerticalFlip(p = 1),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomVerticalFlip)
		#2->RandomRotation
        RandomRotation = transforms.Compose([
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.RandomRotation(360),
            transforms.Pad(padding, padding_mode='edg'),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomRotation)
		#3->RandomResizedCrop
        RandomResizedCrop = transforms.Compose([
            transforms.Resize(256),  # mandate
            transforms.RandomResizedCrop(size = 224),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomResizedCrop)
		#4->RandomPerspective
        RandomPerspective = transforms.Compose([
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.RandomPerspective(distortion_scale=0.4, p = 1),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomPerspective)

        #photometric transform
        #0->GaussianBlur
        #1->RandomAdjustSharpness
        GaussianBlur = transforms.Compose([#?torch.nn.Sequential
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor()])  # mandate
        transform_list.append(GaussianBlur)

        RandomAdjustSharpness = transforms.Compose([#?torch.nn.Sequential
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.RandomAdjustSharpness(sharpness_factor = 2, p=1),
            transforms.ToTensor()])  # mandate
        transform_list.append(RandomAdjustSharpness)

        normalize = transforms.Normalize(mean=[ 0.499, 0.559, 0.535], std=[0.021, 0.018, 0.019])
        Normalize = transforms.Compose([
            transforms.Resize(224),  # mandate
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
            normalize
            ])
		#select the transform method according to the input "mode"
        mode_transform =  transforms.Resize(224)
        if self.mode == 7:
            print("random choice")
            mode_transform = transforms.RandomChoice(transform_list)
        else:
            print("select method:",self.mode)
            mode_transform = transform_list[self.mode]
        if self.mode == 8:
            print("Normalize")
            mode_transform = Normalize
        #
        train_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'train'),
                                  self.dataset_xml_list,
                                  transform = mode_transform
                                  )
        val_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'val'),
                                self.dataset_xml_list,
                                transform=transforms.Compose([transforms.Resize(patch_size),
                                                             transforms.CenterCrop(patch_size),
                                                             transforms.ToTensor()
                                                             #normalize
                                                             ]))
        test_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'test'),
                                 self.dataset_xml_list,
                                 transform=transforms.Compose([transforms.Resize(patch_size),
                                                              transforms.CenterCrop(patch_size),
                                                              transforms.ToTensor()
                                                              #normalize
                                                              ]))
        return train_set, val_set, test_set

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): mini batch size in data loader
            workers (int): number of parallel cpu threads for data loading
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        
        Returns:
            train_loader, val_loader, test_loader of type torch.utils.data.DataLoader
        """
        train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=workers, batch_size=batch_size,
                                                   shuffle=True, pin_memory=is_gpu)
        val_loader = torch.utils.data.DataLoader(self.val_set, num_workers=workers, batch_size=batch_size,
                                                 shuffle=False, pin_memory=is_gpu)
        test_loader = torch.utils.data.DataLoader(self.test_set, num_workers=workers, batch_size=batch_size,
                                                  shuffle=False, pin_memory=is_gpu)

        return train_loader, val_loader, test_loader
