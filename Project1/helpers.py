import numpy as np

"""
data processing related methods
* data preprocessing
* train/test split
* cross-validation data split
"""

def build_label(data):
    """extract label from original data, second column is label by default
    
    :param data: original data from csv
    """
    label = data[:,1]
    dict_x = {'s': 1, 'b': -1} 
    for i in dict_x.keys():
        label[label==i]=dict_x.get(i)
    return label.astype('float64')

def data_norm(x):
    for i in range(0, x.shape[1]):
        mean = np.mean(x[:,i])
        std = np.std(x[:,i])
        x[:,i] = (x[:,i]-mean) / std
    return x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

		:param x: input data of shape (N, D)
		:param degree: maximum degree for expansion
		:return matrix: expanded data of shape [sample, degree]
		"""
    # polynomial basis function:
    poly_x = []
    for j in range(1, degree+1):
        poly_x.append(x**j)
    poly_x.append(x[:,0].reshape(-1,1)**0)  # add 1 column

    return np.concatenate(poly_x, axis=1)  # shape [sample, degree]

def build_cross(x):
    """Build the cross multiplication term feature to the input

    :param x: input x with shape (N,D)
    """
    cross_x = []
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if(i != j):
                cross_x.append((x[:,i]*x[:,j]).reshape(x.shape[0],1))
    return np.concatenate(cross_x, axis=1) 

def build_sqrt(x):
    """Expand the input with absolute sqrt operation

    :param x: input x with shape (N,D)
    """
    sqrt_x = x
    sqrt_x[x>0] = np.sqrt(x[x>0])
    sqrt_x[x==0] = 0
    sqrt_x[x<0] = (-1)*np.sqrt((-1)*x[x<0])

    return sqrt_x

def build_log(x):
    """
    Expand the input with absolute log operation
    
    :param x: input x with shape (N,D)
    """
    log_x = x
    log_x[x>0] = np.log(x[x>0])
    log_x[x==0] = 0
    log_x[x<0] = (-1)*np.log((-1)*x[x<0])
    return log_x
    

def feature_expansion(x, degree):
    """
    Do feature expansions as:
    1. build poly term
    2. build log term
    3. build square root term
    4. build sin and cos term
    5. build cross multiplication term
    """
    new_features = []
    #build poly
    poly_x = build_poly(x, degree)
    new_features.append(poly_x)
    #build log
    log_x = build_log(x)
    new_features.append(log_x)
    #build square root
    sr_x = build_sqrt(x)
    new_features.append(sr_x)
    #build sin and cos
    sin_x = np.sin(x)
    new_features.append(sin_x)
    cos_x = np.cos(x)
    new_features.append(cos_x)
    #build cross
    cross_x = build_cross(x)
    new_features.append(cross_x)
    return np.concatenate(new_features, axis=1)

def outlier_indexs(x):
    """
    get the indices of outliers
    
    :param x: input x with shape (N,D)
    """
    ##copy, need to be replaced
    Q1, Q3 = np.percentile(x, [25, 75])
    IQR = Q3-Q1
    lower_bound = Q1 - (10 * IQR)
    upper_bound = Q3 + (10 * IQR)
    outlier_index = np.where((x < lower_bound) | (x > upper_bound))
    return outlier_index[0]


def remove_outliers(x, y):
    """
    remove outliers using indices
    """
    delete = []
    for i in range(x.shape[1]):
        indexs = outlier_indexs(x[:,i])
        delete.append(indexs)
    delete = np.concatenate(delete, axis=0)
    delete = np.unique(delete).astype(int)
    return np.delete(x, delete, axis = 0), np.delete(y, delete, axis = 0)

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    
    :param x: input data of shape (N, D)
    :param y: label data of shape (N,)
    :param ratio: ratio of training data
    :return x_train, x_test, y_train, y_test
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]  # Sample num
    shuffle_index = np.random.permutation(N)
    tr_N = int(ratio * N)  # training data num
    tr_index = shuffle_index[:tr_N]  # trainining data index
    te_index = shuffle_index[tr_N:]
    
    x_train, x_test = x[tr_index], x[te_index]
    y_train, y_test = y[tr_index], y[te_index]
    return x_train, x_test, y_train, y_test	
	
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def get_cross_validation_data(y, x, k, degree, seed, k_fold=10):
    """return the cross validation data.
    
    :param k_fold: cross validation fold number
    :param k: k-th fold for validation
    :param degree: maximum degree for expansion
    """
    # get k indices
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    # get k'th subgroup in test, others in train:
    tr_indices = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k], axis=0)
    te_indices = k_indices[k]
    x_te, y_te = x[te_indices], y[te_indices]
    x_tr, y_tr = x[tr_indices], y[tr_indices]

    # form data with polynomial degree:
    x_tr = feature_expansion(x_tr, degree)
    x_te = feature_expansion(x_te, degree)
    
    return x_tr, x_te, y_tr, y_te	

	
"""
Model optimization related methods
* loss functions
* gradient functions
"""

def compute_gradient(y, tx, w):
    """Calculate the gradient.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: gradient
    """
    e = y - tx.dot(w).squeeze()
    grad = -tx.T.dot(e)/len(e)
    return grad

def sigmoid(x):
    """Calculate Sigmoid
    :param xt: input data of shape (N, 1)
    :param w: model weights of shape (1, D)
    :return: sigmoid
    """

    return 1.0 / (1 + np.exp(-x))

def compute_loss_lr(y, tx, w):
    """compute the cost by negative log likelihood. 
    cost = -Y'log(H) - (1 - Y')log(1 - H)

    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    """
    pred = sigmoid(tx.dot(w)).squeeze()
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_gradient_lr(y, tx, w):
    """Calculate the gradient.

    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    """
    pred = sigmoid(tx.dot(w)).squeeze()
    gradient = tx.T.dot(pred - y)
    return gradient

def compute_gradient_rlr(y, xt, w, lambda_):
    """Calculate the gradient.
    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: gradient
    """
    gradient_lr = compute_gradient_lr(y, xt, w)
    penalty = lambda_ * w
    return gradient_lr + penalty

def compute_loss_rlr(y, xt, w, lambda_):
    """compute the cost by negative log likelihood with L2
	cost = -Y'log(H) - (1 - Y')log(1 - H) + Lambda/2*W**2
	:param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MSE loss for regular logistic regression
	"""
    loss_lr = compute_loss_lr(y, xt, w)
    penalty = (lambda_ / 2) * sum(w ** 2)
    return loss_lr + penalty

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param batch_size: the size of batch
	:param num_batch: the number of batches
	:param shuffle: whether or not shuffle is needed 
    :return: a batch of data	
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mse_loss(y, tx, w):
    """Calculate the Mean Square Error loss.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MSE loss
    """
    e = y - tx.dot(w).squeeze()
    loss = e.dot(e) / (2 * len(e))

    return loss

def mae_loss(y, tx, w):
    """Calculate the Mean Absolute Error loss.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MAE loss
    """
    e = y - tx.dot(w).squeeze()
    loss = np.sum(np.abs(e)) / (2 * tx.shape[0])

    return loss

def get_accuracy(y_pred, y_gt):
    """
    Get the accuracy of predictions based on the ground-truth
    """
    return (y_pred == y_gt).sum() / y_gt.shape[0]

