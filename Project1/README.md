## Machine Learning - Project 1 (Team: SeaStar)

In this repository, you can find our work for the Project 1 of the [Machine Learning](https://github.com/epfml/ML_course) at [EPFL](http://epfl.ch). The background of the project could be found [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf).

We took part in the [competion](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards) and got an accuracy of 83.1%.

You can find our report in `report.pdf` and our code in `Project1` folder.

The following information mainly focuses on explaining our code.

First, ensure that you put `train.csv` and `test.csv` in the `data` folder at the root of `Project1` folder.

Then, you can run `run.py` to create `submit.csv`, the output of our model, which provides the prediction on `test.csv`.

### `helpers.py`
The following methods are for data processing and model optimazation:
#### `Data Preprocessing`
- **`build_label`**: extract label from original data.
- **`data_norm`**: Normalize the input data.
- **`build_poly`**: Polynomial basis functions for input data x, for j=0 up to j=degree.
- **`build_cross`**: Cross multiply the columns of the input data and return the result.
- **`build_sqrt`**: Caculate the square root value of every elements of the input data and return the result.
- **`build_log`**: Caculate the log value of every elements of the input data and return the result.
- **`feature_expansion`**: Expand features of input data by applying series of different arithmetic operations. 
- **`outlier_indexs`**: Return indexs of outliers of the input data.
- **`remove_outliers`**: Remove the outliers of the input data.
- **`feature_expansion`**: Expand features of input data by applying series of different arithmetic operations. 
#### `Train/Test Split`
- **`split_data`**: Split the dataset based on the split ratio to get train subset and test subset from original train set.
#### `Cross-Validation Data Split`
- **`build_k_indices`**: Build k indices for k-fold.
- **`get_cross_validation_data`**: Return the cross validation data.
#### `Model Optimalization`
- **`compute_gradient`**: Calculate the gradient.
- **`sigmoid`**: Calculate sigmoid.
- **`compute_loss_lr`**: Compute the cost by negative log likelihood.
- **`compute_gradient_lr`**: Calculate the gradient.
- **`compute_gradient_rlr`**: Calculate the gradient.
- **`compute_loss_rlr`**: Compute the cost by negative log likelihood with L2
- **`batch_iter`**: Generate a minibatch iterator for a dataset.
- **`mse_loss`**: Calculate the Mean Square Error loss.
- **`mae_loss`**: Calculate the Mean Absolute Error loss.
- **`get_accuracy`**: Get the accuracy of predictions based on the ground-truth

### `implementations.py`
Six models implemented:
- **`least_squares_GD`**: Perform gradient descent.
- **`least_squares_SGD`**: Perform Stochastic gradient descent algorithm.
- **`least_squares`**: Calculate the least squares solution.
- **`ridge_regression`**: Execute ridge regression.
- **`logistic_regression`**: Perform logistic regression.
- **`reg_logistic_regression`**: Perform regularized logistic regression.

### `proj1_helpers.py`
The original helpers provided in project 1:
- **`load_csv_data`**: Loads data and returns y (class labels), tX (features) and ids (event ids).
- **`predict_labels`**: Generates class predictions given weights, and a test data matrix.
- **`create_csv_submission`**: Creates an output file in .csv format for submission to Kaggle or AIcrowd.

### `run.py`
Contain `train` function and multi steps of data pre-processing.

### `experiments.ipynb`
This file contains the code of our experiments, which could reproduce the resuls shown in the report, including tables and figures. 
