import numpy as np
import sys
import tensorflow as tf
import os
from datetime import datetime as dt
from numpy import random as rand
import pickle
import time

def shape(tensor):
    return tuple([d.value for d in tensor.get_shape()])

#Gets a fully connected neural network layer
def fully_connected_layer(in_tensor,out_units,
                        activation_function=tf.nn.relu):
    _, num_features = shape(in_tensor)
    W = tf.get_variable("weights",[num_features,out_units],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable("biases",[out_units],
                    initializer=tf.constant_initializer(0.1))
    return activation_function(tf.matmul(in_tensor,W)+b)

#We can define scopes for our variables. Useful for adversarial stuff.
#This method returns the names of variables used in scope
def variables_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=scope_name)

def oneHot(myList):
	N = len(myList)
	vals = np.unique(myList)
	nVals = len(vals)
	out = np.zeros((N,nVals))
	for i in range(nVals):
		out[myList==vals[i],i] = 1.0
	return out,vals

def getIdxs(myList,vals):
	N = len(myList)
	out = np.zeros(N)
	for i in range(N):
		out[i] = np.where(myList[i]==vals)[0][0]
	return out

def createOH(myList,vals):
	N = len(myList)
	nVals = len(vals)
	out = np.zeros((N,nVals))
	for i in range(N):
		out[i,myList[i]] = 1
	return out
