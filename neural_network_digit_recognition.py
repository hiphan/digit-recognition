import math
import numpy as np
import activation_func as actf
import matplotlib.pyplot as plt

from kaggle import write_output
#  from sklearn.datasets import load_digits


# constant variables about the architecture of the neural network
NUM_LAYER = 4  # including input layer
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_1_SIZE = 25
HIDDEN_LAYER_2_SIZE = 25
NUM_LABELS = 10
LAYER_SIZE = [INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, NUM_LABELS]


# divide pixel values by 255 to obtain values between 0 and 1
def scale_pixels(data):
    return data / 255


# random initialization of thetas for leaky_relu activation function
def random_initial_weight(connection_in, connection_out):
    epsilon = np.sqrt(2) / np.sqrt(connection_in)
    return np.random.randn(connection_out, connection_in + 1) * 2 * epsilon - epsilon


def create_mini_batches(x, y, batch_size):
    m, n = x.shape
    batches = []
    perm = list(np.random.permutation(m))
    shuffled_x = x[perm, :]
    shuffled_y = y[perm, :]
    num_complete_batches = math.floor(m / batch_size)
    last_batch_size = m - batch_size * num_complete_batches
    for i in range(num_complete_batches):
        batch_x = shuffled_x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = shuffled_y[i * batch_size:(i + 1) * batch_size, :]
        batches.append((batch_x, batch_y))
    if last_batch_size > 0:
        batch_x = shuffled_x[batch_size * num_complete_batches:, :]
        batch_y = shuffled_y[batch_size * num_complete_batches:, :]
        batches.append((batch_x, batch_y))
    return batches


def feedfoward_prop(x, theta):
    cache_a = {}
    cache_z = {}
    m, n = np.shape(x)  # m is the number of training samples, n is the number of features
    cache_a[0] = np.append(np.ones((m, 1)), x, axis=1)
    for l in range(1, NUM_LAYER):
        if l < NUM_LAYER - 1:
            cache_z[l] = np.dot(cache_a[l - 1], theta[l].T)
            cache_a[l] = actf.leaky_relu(cache_z[l])
            cache_a[l] = np.append(np.ones((m, 1)), cache_a[l], axis=1)
        else:
            cache_z[l] = np.dot(cache_a[l - 1], theta[l].T)
            cache_a[l] = actf.softmax(cache_z[l])
    return cache_a, cache_z


def cost_function(x, y, lambd, theta, cache_a):
    m, n = np.shape(x)
    y_truth = np.zeros((m, NUM_LABELS))
    for i in range(m):
        y_truth[i, int(y[i])] = 1
    # print(1 - cache_a[NUM_LAYER - 1])
    loss_cost = (1 / m) * np.sum(np.multiply(- y_truth, np.log(cache_a[NUM_LAYER - 1])))
                       # - np.multiply(1 - y_truth, np.log(1 - cache_a[NUM_LAYER - 1]))) / m
    reg_cost = 0
    for l in range(1, NUM_LAYER):
        reg_cost += lambd / (2 * m) * np.sum(np.power(theta[l], 2))
    cost_j = loss_cost + reg_cost
    return cost_j


def back_prop(x, y, lambd, theta, cache_a, cache_z):
    m, n = np.shape(x)
    cache_d = {}
    theta_grad = {}
    y_truth = np.zeros((m, NUM_LABELS))
    for i in range(m):
        y_truth[i, int(y[i])] = 1
    cache_d[NUM_LAYER - 1] = cache_a[NUM_LAYER - 1] - y_truth
    theta_grad[NUM_LAYER - 1] = np.dot(cache_d[NUM_LAYER - 1].T, cache_a[NUM_LAYER - 2]) / m + lambd / m \
                               * np.sum(theta[NUM_LAYER - 1][:, 1:])
    for l in range(NUM_LAYER - 2, 0, -1):
        cache_d[l] = np.dot(cache_d[l + 1], theta[l + 1])
        cache_d[l] = cache_d[l][:, 1:]
        cache_d[l] = np.multiply(cache_d[l], actf.leaky_relu_gradient(cache_z[l]))
        theta_grad[l] = np.dot(cache_d[l].T, cache_a[l - 1]) / m + lambd / m * np.sum(theta[l][:, 1:])
    return theta_grad


def update_theta(theta, theta_grad, learning_rate):
    for l in range(1, NUM_LAYER):
        theta[l] = theta[l] - learning_rate * theta_grad[l]
    return theta


# mini-batch gradient descent
def nn_model(x, y, lambd=0.09, batch_size=128, num_epochs=1000, learning_rate=0.0006):
    theta = {}
    cost_history = []
    mini_batches = create_mini_batches(x, y, batch_size)

    # randomly initialize theta matrices
    for i in range(1, NUM_LAYER):
        theta[i] = random_initial_weight(LAYER_SIZE[i - 1], LAYER_SIZE[i])

    for k in range(num_epochs):
        for t in range(len(mini_batches)):
            (curr_batch_x, curr_batch_y) = mini_batches[t]
            cache_a, cache_z = feedfoward_prop(curr_batch_x, theta)
            cost = cost_function(curr_batch_x, curr_batch_y, lambd, theta, cache_a)
            if t % 10 == 0:
                cost_history.append(cost)
            if k % 100 == 99 and t == len(mini_batches) - 1:
                print("Cost after " + str(k + 1) + " epochs: " + str(cost))
            theta_grad = back_prop(curr_batch_x, curr_batch_y, lambd, theta, cache_a, cache_z)
            theta = update_theta(theta, theta_grad, learning_rate)
    return theta, cost_history


def load_data():
    data = np.genfromtxt('train.csv', delimiter=',', skip_header=1, dtype=float)
    x = data[:, 1:785]
    x_train = scale_pixels(x)
    y = data[:, 0]
    y_train = y.reshape((y.shape[0], 1))
    return x_train, y_train


def load_test():
    x_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1, dtype=float)
    x_test = scale_pixels(x_test)
    return x_test


# predict digit with the final weights
def predict(x, theta):
    m, n = np.shape(x)
    cache_a, cache_z = feedfoward_prop(x, theta)
    predictions = np.argmax(cache_a[NUM_LAYER - 1], axis=1)
    predictions = predictions.reshape((m, 1))
    return predictions


# accuracy of predictions for the training set
def training_accuracy(pred, y):
    m = np.shape(y)[0]
    tt = 0
    comparision = np.equal(pred, y)
    for i in range(m):
        if comparision[i]:
            tt += 1
    return tt / m


if __name__ == "__main__":
    x_train, y_train = load_data()
    x_test = load_test()
    theta_final, cost_hist = nn_model(x_train, y_train)
    y_hat = predict(x_train, theta_final)
    print("Training accuracy: " + str(training_accuracy(y_hat, y_train) * 100) + " %")
    plt.plot(cost_hist)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations for lr = 0.003, lambda = 0.01, mini-batch size = 128, num_epochs = 500')
    plt.savefig('cost-vs-iteration-1.png')
    plt.show()

    # predictions of the test data -> kaggle
    y_hat_test = predict(x_test, theta_final)
    write_output(y_hat_test, "predictions.csv")
