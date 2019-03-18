import numpy as np
import activation_func as actf
import matplotlib.pyplot as plt
#  from sklearn.datasets import load_digits


# constant variables containing information of the neural network
NUM_LAYER = 4  # including input layer
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_1_SIZE = 25
HIDDEN_LAYER_2_SIZE = 25
NUM_LABELS = 10
LAYER_SIZE = [INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, NUM_LABELS]
np.random.seed(1)


# divide pixel values by 255 to obtain values between 0 and 1
def scale_pixels(data):
    return data / 255


# random initialization of thetas for leaky_relu activation function
def random_initial_weight(connection_in, connection_out):
    epsilon = np.sqrt(2) / np.sqrt(connection_in)
    return np.random.randn(connection_out, connection_in + 1) * 2 * epsilon - epsilon


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
            cache_a[l] = actf.sigmoid(cache_z[l])
    return cache_a, cache_z


def cost_function(x, y, lambd, theta, cache_a):
    m, n = np.shape(x)
    y_truth = np.zeros((m, NUM_LABELS))
    for i in range(m):
        y_truth[i, int(y[i])] = 1
    loss_cost = np.sum(np.multiply(- y_truth, np.log(cache_a[NUM_LAYER - 1])) - np.multiply(1 - y_truth, np.log(
        1 - cache_a[NUM_LAYER - 1]))) / m
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


def nn_model(x, y, lambd=0.01, num_iterations=10000, learning_rate=0.1):
    theta = {}
    cost_history = []
    for i in range(1, NUM_LAYER):
        theta[i] = random_initial_weight(LAYER_SIZE[i - 1], LAYER_SIZE[i])
    for k in range(num_iterations):
        cache_a, cache_z = feedfoward_prop(x, theta)
        cost = cost_function(x, y, lambd, theta, cache_a)
        cost_history.append(cost)
        if k % 1000 == 0:
            print("Cost " + str(k) + ": " + str(cost))
        theta_grad = back_prop(x, y, lambd, theta, cache_a, cache_z)
        update_theta(theta, theta_grad, learning_rate)
    return theta, cost_history


def load_data():
    data = np.genfromtxt('train.csv', delimiter=',', skip_header=1, dtype=float)
    x = data[:, 1:785]
    x_train = scale_pixels(x)
    y = data[:, 0]
    y_train = y.reshape((y.shape[0], 1))
    return x_train, y_train


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
    x, y = load_data()
    theta_final, cost_hist = nn_model(x, y, num_iterations=5000)
    y_hat = predict(x, theta_final)
    print("Training accuracy: " + str(training_accuracy(y_hat, y) * 100) + "%")
    plt.plot(cost_hist)
    plt.show()
