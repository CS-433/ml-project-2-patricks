# Machine Learning - Project 2 (Team: SeaStar)

In this repository, you can find our work for the Project 2 of the [Machine Learning](https://github.com/epfml/ML_course) at [EPFL](http://epfl.ch). We focus on the crack concrete classification problem as described [here](https://zenodo.org/record/2620293#.YZTqbr3MJqt), with the CODEBRIM dataset provided.


## Contribution

1. We evaluate and compare expert designed and NAS generated CNN architectures for the multi-target concrete defect classification task, and obtain highly competitive result using ZenNAS with much less parameters compared to the expert-designed models.
2. Further, We show that cross-domain transfer learning could greatly boost the model performance, under which the ZenNAS model achieves best multi-target accuracy and surpassed the best result from MetaQNN in the original CODEBRIM paper (from 72.2\% to 75.6\%) 
3. We validate the performance gain using Grad-CAM to inspect attention pattern of last few convolutional layers, which shows that our transferred models' attention is better aligned with the defect area.


## File structure of our project

.
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
│   ├── EfficientNet.ipynb\
│   ├── Image_banlance.pkl\
│   ├── NAS_result_analysis.ipynb\
│   ├── ZenNas_example.py\
│   ├── __init__.py\
│   ├── copy_balance.ipynb\
│   ├── datasets.py\
│   ├── focal_loss.py\
│   ├── model_acc_param.csv\
│   ├── pytorch_grad_cam\
│   ├── result_analysis.ipynb\
│   ├── result_comparison.html\
│   ├── result_comparison.png\
│   └── test_image_trans.ipynb\
├── README.md\
├── run_sample.ipynb\
├── pytorch_grad_cam\
├── ref_codes\
├── cam_test\
└── train_log\


### `run_sample.ipynb`

A simple workflow using **ZenNAS-1** model as an example to get a quick overview of our project.

### `sample`

Include five sample pictures and labels 

