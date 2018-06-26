# standard python libraries
import os
import sys
import logging

## 3rd part libraries
import pandas as pd
import numpy as np

## keras utilities
import keras
from keras.layers import (
	Dense,
	Activation
	)
from keras.models import Model, Sequential

# custom libraries
from module_constants import *
import utils


def define_deep_nn_model(n_hidden_nodes_list, num_classes, input_shape):
	model = Sequential()
	model.add(Dense(n_hidden_nodes_list[0], input_dim = input_shape))
	model.add(Activation('tanh'))
	for n_nodes in range(1,len(n_hidden_nodes_list)):
		model.add(Dense(n_hidden_nodes_list[n_nodes]))
		model.add(Activation('relu'))

	model.add(Dense(num_classes))
	model.add(Activation('sigmoid'))

	return model

def caller ():
	
	x_train, y_train, x_test, y_test, num_classes, input_shape = utils.load_mnist_data()

	model = define_deep_nn_model([64, 128, 256], num_classes, input_shape)

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))

	## predict on test set 
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == "__main__":
	caller()
	