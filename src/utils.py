import numpy as np

# load mnist dataset
import keras
from keras.datasets import mnist

def load_mnist_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2]).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2]).astype('float32')
	x_train/= 255.0
	x_test /= 255.0
	num_classes = len(np.unique(y_train))
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	input_shape = x_train.shape[1]
	return x_train, y_train, x_test, y_test, num_classes, input_shape