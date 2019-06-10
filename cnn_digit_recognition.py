import h5py
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from kaggle import write_output


def scale_pixels(data):
    return data / 255


def load_train():
    data = np.genfromtxt('train.csv', delimiter=',', skip_header=1, dtype=float)
    x = data[:, 1:785]
    x_train = scale_pixels(x)                   # normalize the train data
    x_train = x_train.reshape(-1, 28, 28, 1)
    m = x_train.shape[0]
    y = data[:, 0].astype(int)
    y_train = np.zeros((m, 10))
    y_train[np.arange(m, dtype=int), y] = 1

    # Split the training data into train and cross-validation set. Train: 36960, CV: 5040
    x_train, x_cross_vali, y_train, y_cross_vali = train_test_split(x_train, y_train, test_size=0.12, stratify=y_train)
    return x_train, x_cross_vali, y_train, y_cross_vali


def load_test():
    x_test = np.genfromtxt('test.csv', delimiter=',', skip_header=1, dtype=float)
    x_test = scale_pixels(x_test)               # normalize the test data
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_test


# 6 layers: Input (28, 28, 1) -> 50 -> 100 -> 500 -> 1000 -> FC -> 10 -> 10
def digitRecognizerModel(input_shape):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(filters=50, kernel_size=(7, 7), padding='same', input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(filters=100, kernel_size=(5, 5), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(filters=500, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(filters=1000, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))0
    model.add(keras.layers.Dense(10, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='he_normal'))

    return model


X_train, X_cross_vali, Y_train, Y_cross_vali = load_train()
X_test = load_test()
digitModel = digitRecognizerModel((28, 28, 1))
digitModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
digitModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=64, validation_data=(X_cross_vali, Y_cross_vali), verbose=2)

digitModel.save('Result/cnn_model.h5')

prediction_probability = digitModel.predict(X_test)
prediction_label = np.argmax(prediction_probability, axis=1)
write_output(prediction_label, 'Result/cnn_predictions.csv')

# 0.98114
