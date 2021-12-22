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
    import chardet
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

def show_label(label, is_test = False):
    '''
    show sample label
    '''
    # from tabulate import tabulate
    label = label[0]
    # d = [ ["Background", label[0]],
    #  ["Crack", label[1]],
    #  ["Spallation", label[2]],
    #  ["Efflorescence", label[3]],
    #  ["ExposedBars", label[4]],
    #  ["CorrosionStain", label[5]]]
    # print(tabulate(d, headers=["Sort", "0/1"]))

    label_list = ["Background", "Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
    if is_test:
        print("The pridiction is/are: ")
    else:
        print("The ground truth is/are: ")
    for i in range(len(label)):
        if label[i] == 1:
            print(label_list[i])
    

def show_sample(data):
    '''
    show sample piture

    :param data: tensor object picture data
    '''
    import torch
    from torchvision.transforms import ToPILImage
    show = ToPILImage() 
    show(data[0]).show()

def get_random_sample():
    """
    load a random sample for test
    
    return data, label
    """
    import datasets as DS
    class Args:
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


def show_attenttion(model, data):
    from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
    from pytorch_grad_cam import GuidedBackpropReLUModel
    from pytorch_grad_cam.utils.image import show_cam_on_image, \
        deprocess_image, \
        preprocess_image
    import torchvision.transforms as transforms
    import cv2
    import numpy as np

    methods = \
        {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad}

    toPILTransform = transforms.ToPILImage()
    rgb_img = toPILTransform(data[0])#select the first image
    rgb_img = np.float32(rgb_img) / 255


    target_layers = [model._modules['module_list'][5]]
        
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods["gradcam++"]
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=False) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1

        grayscale_cam = cam(input_tensor=data,
                            target_category=target_category,
                            aug_smooth=True,
                            eigen_smooth=True)
        
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda= False)
    gb = gb_model(data, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    from PIL import Image 
    image = Image.fromarray(cam_image)
    image.show() 

def get_sample(path = None):
    if not path:
        return get_random_sample()
    else:
        from PIL import Image
        from torchvision import transforms
        import xml.etree.ElementTree as ElementTree
        import numpy as np
        xml_path = "./sample/defects.xml"
        transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        img = Image.open(path)
        file_name = path.split("/")[-1]
        img_ = transformer(img).unsqueeze(0)
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
        for defect in root:
            crop_name = list(defect.attrib.values())[0]
            if file_name == crop_name:
                out = np.zeros(6, dtype=np.float32)
                for i in range(6):
                    if defect[i].text == '1':
                        out[i] = 1.0
                break

        return img_, torch.from_numpy(out).unsqueeze(0)

