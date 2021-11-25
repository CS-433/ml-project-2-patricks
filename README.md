# Machine Learning - Project 2 (Team: SeaStar)

In this repository, you can find our work for the Project 2 of the [Machine Learning](https://github.com/epfml/ML_course) at [EPFL](http://epfl.ch). We focus on the crack concrete classification problem as described [here](https://zenodo.org/record/2620293#.YZTqbr3MJqt), with the CODEBRIM dataset provided.

---
## Darts for CODEBRIM Neural Architecture Search


### Requirements
`Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0`

To run the Neural Architecture Search with Darts, simply type
`cd Darts && python train_search.py --unrolled --batch_size 16`

In our case, a 16GB RTX5000 GPU would allow `batch_size=2`(which is small due to the large image size (224x224) of CODEBRIM dataset), so you can decrease the batch_size of CUDA out of memory :)
