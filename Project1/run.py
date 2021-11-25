from numpy.core.fromnumeric import size
from helpers import get_cross_validation_data, compute_loss_rlr, \
 mse_loss, sigmoid, build_poly, split_data, mae_loss, data_norm, feature_expansion, remove_outliers, get_accuracy
from implementations import ridge_regression
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels

import numpy as np

def train(X, y, degree, lambda_):
    """train the ridge regression model using full dataset
    
    :param tx: input data of shape (N, D)
    :param y: label data of shape (N,)
    :param degree: degree for build_poly, expand features
    :param lambda_: weight of penalty term
    :return: w: final weight vector
    """
    seed = 2021
    # Use the grid-searched hyper-parameters for full dataset training
    x_tr, x_te, y_tr, y_te = split_data(X, y, 1, seed=seed)
    x_tr = feature_expansion(x_tr, degree)
    x_te = feature_expansion(x_te, degree)

    # model parameter initialization
    w = np.random.randn(x_tr.shape[1], 1)

    w, loss = ridge_regression(y_tr, x_tr, lambda_)

    return w, loss

# set the random seed
np.random.seed(2021)

# grid-searched best hyper-parameters for each of the subgroup model
degrees = [7, 10, 11]
lambdas = [1e-5, 1e-5, 1e-5]

#load data and split it according to the column named PRI_ject_num
y, X, ids = load_csv_data("data/train.csv") 
kind = X[:,-8]

#get index set of different PRI_ject_num(0, 1, 2&3)
set0 = np.where(kind == 0)
set1 = np.where(kind == 1)
set2_3 = np.where((kind == 2)|(kind == 3))

#delete the columns that are meaningless or uncomputable based on specific PRI_ject_num
#collect data sets(specific row groups) according to different PRI_ject_num
zero_delete_col = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29] 
one_delete_col = [4, 5, 6, 12, 22, 26, 27, 28]

X0 = np.delete(X, zero_delete_col, axis = 1)[set0,:].squeeze()
y0 = y[set0]
X1 = np.delete(X, one_delete_col, axis = 1)[set1,:].squeeze()
y1 = y[set1]
X2_3 = X[set2_3,:].squeeze()
y2_3 = y[set2_3]


#train different models based on different PRI_ject_num
w0, loss1 = train(X0, y0, degrees[0], lambdas[0])
w1, loss2 = train(X1, y1, degrees[1], lambdas[1])
w2_3, loss3 = train(X2_3, y2_3, degrees[2], lambdas[2])
print(f'loss for subset 1 is {loss1}')
print(f'loss for subset 2 is {loss2}')
print(f'loss for subset 3 is {loss3}')

print('\n finish training, start predicting results ...')

#make predictions based on different models and concact them
y, X_test, ids_test = load_csv_data("data/test.csv") 

#get index set of different PRI_ject_num(0, 1, 2&3)
kind = X_test[:,-8]
set0 = np.where(kind == 0)
set1 = np.where(kind == 1)
set2_3 = np.where((kind == 2)|(kind == 3))

#delete the columns that are meaningless or uncomputable based on specific PRI_ject_num
#collect data sets(specific row groups) according to different PRI_ject_num
X0 = np.delete(X_test, zero_delete_col, axis = 1)[set0,:].squeeze()
X0 = feature_expansion(X0, degrees[0])
y[set0] = np.dot(X0, w0)

X1 = np.delete(X_test, one_delete_col, axis = 1)[set1,:].squeeze()
X1 = feature_expansion(X1, degrees[1])
y[set1] = np.dot(X1,w1)

X2_3 = X_test[set2_3,:].squeeze()
X2_3 = feature_expansion(X2_3, degrees[2])
y[set2_3] = np.dot(X2_3,w2_3)

#create labels
y[np.where(y <= 0)] = -1
y[np.where(y > 0)] = 1

create_csv_submission(ids_test, y, "submit.csv")
print('\nResult generated as submit.csv!')

