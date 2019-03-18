import numpy as np


# compute the sigmoid / logistic function for z
def sigmoid(z):
    return 1 / (1 + np.exp((-1) * z))


# compute the gradient of the sigmoid function for z
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


# compute the tanh function for z
def tanh(z):
    return np.tanh(z)


# compute the gradient of the tanh function for z
def tanh_gradient(z):
    z_tanh = tanh(z)
    return 1 - np.square(z_tanh)


# relu activation function
def relu(z):
    return np.maximum(0, z)


# compute the gradient of the relu function for z
def relu_gradient(z):
    return np.int_(z > 0)


# leaking relu activation function
def leaky_relu(z):
    return np.maximum(0.01 * z, z)


# compute the gradient of the leaky relu function for z
def leaky_relu_gradient(z):
    return np.int_(z > 0) - 0.01 * np.int_(z < 0)


# softmax activation function
def softmax(z):
    numerator = np.exp(z - np.max(z))
    return numerator / np.sum(numerator)


# compute the gradient of the the softmax function for z
# https://stats.stackexchange.com/questions/215521/how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent
def softmax_gradient(z):
    s_rs = softmax(z).reshape(z.shape[0], 1)
    return np.diagflat(s_rs) - np.dot(s_rs, s_rs.T)
