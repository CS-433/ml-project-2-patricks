# Machine Learning - Project 2 (Team: SeaStar)

In this repository, you can find our work for the Project 2 of the [Machine Learning](https://github.com/epfml/ML_course) at [EPFL](http://epfl.ch). We focus on the crack concrete classification problem as described [here](https://zenodo.org/record/2620293#.YZTqbr3MJqt), with the CODEBRIM dataset provided.


## Contribution

1. We evaluate and compare expert designed and NAS generated CNN architectures for the multi-target concrete defect classification task, and obtain highly competitive result using ZenNAS with much less parameters compared to the expert-designed models.
2. Further, We show that cross-domain transfer learning could greatly boost the model performance, under which the ZenNAS model achieves best multi-target accuracy and surpassed the best result from MetaQNN in the original CODEBRIM paper (from 72.2\% to 75.6\%) 
3. We validate the performance gain using Grad-CAM to inspect attention pattern of last few convolutional layers, which shows that our transferred models' attention is better aligned with the defect area.


## File structure of our project

.\
├── NAS_designed_model\
│   ├── EfficientNet_grid_search.ipynb\
│   ├── enas.ipynb\
│   ├── train_crack_efficientNet.ipynb\
│   └── train_crack_zenNAS.ipynb\
├── data_augmentation\
│   ├── data_analysis.ipynb\
│   ├── data_augmentation.ipynb\
│   └── datasets.py\
├── expert_designed_model\
│   ├── train_crack_resnet.ipynb\
│   └── train_crack_vgg.ipynb\
├── model_scripts\
│   ├── hard_ZenNas_withPretrain.pth\
│   ├── hard_ZenNas_withoutPretrain.pth\
│   ├── zennet_imagenet1k_flops400M_res224.txt\
│   ├── zennet_imagenet1k_flops600M_res224.txt\
│   └── zennet_imagenet1k_flops900M_res224.txt\
├── sample\
│   ├── defects\
│   └── defects.xml\
├── tools\
│   ├── Image_banlance.pkl\
│   ├── NAS_result_analysis.ipynb\
│   ├── ZenNas_example.py\
│   ├── copy_balance.ipynb\
│   ├── datasets.py\
│   ├── focal_loss.py\
│   ├── model_acc_param.csv\
│   ├── pytorch_grad_cam\
│   ├── result_analysis.ipynb\
│   ├── result_comparison.html\
│   ├── result_comparison.png\
├── README.md\
├── run_sample.ipynb\
├── ref_codes\
├── cam_test\
└── train_log\


### `run_sample.ipynb`

A simple workflow using **ZenNAS-1** model as an example to get a quick overview of our project.

### `sample`

Include five sample pictures and labels 

### `NAS_designed_model`

Notebook scripts for training models of [ENAS](http://proceedings.mlr.press/v80/pham18a/pham18a.pdf), [ZenNas](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Zen-NAS_A_Zero-Shot_NAS_for_High-Performance_Image_Recognition_ICCV_2021_paper.pdf), and [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf). Also includes a script for conducting a grid search on hyper-parameters batchsize, patchsize, weight decay using EfficientNet.

### `Expert designed model`

Notebook scripts for training models of [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/), [VGG](https://pytorch.org/hub/pytorch_vision_vgg/), [AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/), [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/).

### `Data Augmentation`

*data_analysis.ipynb* includes our code for analyzing data distribution and resolution.\
*data_augmentation.ipynb* includes 9 methods that we used to transform the data, such as RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomResizedCrop, RandomPerspective, GaussianBlur, RandomAdjustSharpness, random_select, normalize.\
*datasets.py* assists in reading data files.

### `model_scripts`

Including model structure produced by ZenNas method and parameter weight files.

### `tools`

*NAS_result_analysis.ipynb* is used for analyzing the result of NAS like Agent Searching Reward.\
*ZenNas_example* includes functions for drawing and generating results in run_sample.ipynb.\
*copy_balance.ipynb* is over-sampling method to address the problem of class imbalance.\
*focal_loss.py* implements focal loss to address the problem of class imbalance.\
*pytorch_grad_cam* is an adjusted python package implementing [Grad_Cam](http://gradcam.cloudcv.org/).\
*result_analysis* summarizes results and analyzes comprehensively with the accuracy rate and the number of model parameters.\

### `ref_codes`

Reference code from github etc. Like the code from the original paper and ZenNas code.

### `train_log`

Part of our training log.
