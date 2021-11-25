import numpy as np
from helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param initial_w: initial weight vector of shape (1, D)
    :param max_iters: Maximum number of iterations
    :param gamma: step-size
    :return: (w, loss): final weight vector and loss value
    """
    w = initial_w
    for _ in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = mse_loss(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad
    print(loss)
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Perform Stochastic gradient descent algorithm.
    
    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param initial_w: initial weight vector of shape (1, D)
    :param batch_size: the size of batch, 1 for stochastic gradient
    :param max_iters: Maximum number of iterations
    :param gamma: step-size
    :return: (w, loss) including final weight vector and final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            loss = mse_loss(y, tx, w)

            # update w by gradient
            w = w - gamma * grad

    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param loss_func: function object, which loss function to use
    :return: (w, loss): final weight vector and loss value
    """
    if len(y.shape) > 1:
        y = y.squeeze()
    
	# solve normal equation
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

	# calculate the loss
    loss = mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """execute ridge regression

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param lambda_: weight of penalty term
    :return: (w, loss): final weight vector and loss value
    """
    if len(y.shape) > 1:
        y = y.squeeze()

    N = tx.shape[0]  # num of data samples
    I = np.identity(tx.shape[1])
    a = tx.T.dot(tx)+lambda_ * 2 * N * I
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mse_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression.
    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D),eg: np.c_[np.ones((y.shape[0], 1)), x]
    :param initial_w: initial weight vector of shape (D, 1), eg: np.zeros((tx.shape[1], 1))
    :param batch_size: the size of batch, 1 for stochastic gradient
    :param max_iters: Maximum number of iterations
    :param gamma: step-size
    :return: (w, loss) including final weight vector and final loss value
    """
    if len(y.shape) > 1:
        y = y.squeeze()
    w = initial_w.squeeze()
    # start the logistic regression
    for n_iter in range(max_iters):
        w_grad = compute_gradient_lr(y, tx, w)
        w = w - gamma * w_grad
        if (n_iter+1) % 100 == 0:
            loss = compute_loss_lr(y, tx, w)
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter + 1, ti=max_iters, l=loss))
    loss = compute_loss_lr(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm.run
    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D),eg: np.c_[np.ones((y.shape[0], 1)), x]
    :param lambda_: weight of penalty term
    :param initial_w: initial weight vector of shape (D, 1), eg: np.zeros((tx.shape[1], 1))
    :param batch_size: the size of batch, 1 for stochastic gradient
    :param max_iters: Maximum number of iterations
    :param gamma: step-size
    :return: (w, loss) including final weight vector and final loss value
		"""
    if len(y.shape) > 1:
        y = y.squeeze()
    # Define parameters to store w and loss
    w = initial_w.squeeze()
    # start the reg logistic regression
    for n_iter in range(max_iters):
        w_grad = compute_gradient_rlr(y, tx, w, lambda_)
        w = w - gamma * w_grad
        if (n_iter + 1) % 100 == 0:
            loss = compute_loss_rlr(y, tx, w, lambda_)
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter + 1, ti=max_iters, l=loss))
    loss = compute_loss_rlr(y, tx, w, lambda_)
    return w, loss 