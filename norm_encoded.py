'''
Creator:
	Austin "The Man" Talbot
Creation Date:
	7/16/2019
Version history
---------------
Version 1.0
	Only God Himeslf knows
Objects
-------
NMF
	This contains a basic encoded non-negative matrix factorization. Given 
	data where ALL values are non-negative it finds a lower dimensional 
	subspace that models all data as addative sums.
NMFq 
	This contains a non-negative matrix factorization with a smoothness
	penalty. This should ensure that the reconstruction yields meaningful
	features
References
----------
https://www.tensorflow.org/

https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
'''
import numpy as np
import numpy.random as rand
import sys 
import tensorflow as tf
import os
from datetime import datetime as dt
from numpy import random as rand
import pickle
import time
from utils import *
from sklearn import decomposition as dp
from norm_base import NORM_base

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

version = "2.0"

class NMF(NORM_base):
	r'''An encoded non-negative matrix factorization

	Optimizes the following problem

	min_{A,W} ||X-WA(X)||^2 + reg*||A(X)||^2 + reg*||W||^2

	Inherits from
	-------------
	NMF_base

	Attributes added
	----------------
	None

	Methods added
	-------------
	def fit(self,x,wg=None):
		X is our data of interest and wg our our weights for importance
		in reconstruction

	Examples
	--------
	import numpy as np
	import numpy.random as rand
	import tensorflow 
	import pickle,sys,os
	from aNMF import NMF
	N_samples = 10000
	latent_states = 30
	p = 300
	S_true = np.exp(.1*rand.randn(N_samples,latent_states))
	S_test = np.exp(.1*rand.randn(N_samples,latent_states))
	W_true = np.abs(rand.randn(latent_states,p))
	X = np.dot(S_true,W_true)
	X_test = np.dot(S_test,W_true)

	#Pretend that there's two groups
	weights = np.ones(N_samples)
	weights[:N_samples/4] = 4
	weights = weights/np.mean(weights)

	n_comp = 20 #This is the number of states we fit our model with
	model = NMF(n_comp)

	#Fit our model
	model.fit(X,weights)

	#Get the latent scores
	S_train = model.transform(X)
	S_test = model.transform(X_test)

	#Save the features
	model.save_components('AustinIsAwesome.csv')

	References
	----------
	Future JASA paper (hopefully)
	'''
	def __init__(self,n_components,outerIter=300000,LR=1e-5,batchSize=100,
			name='Default_NMF',dirName='./tmp_NMF',device=0,
			percGPU=0.45,encoderActiv='softplus',decoderActiv='softplus',
			trainingMethod='Adam',printIter=1000):
		super(NMF,self).__init__(n_components,outerIter=outerIter,
									LR=LR,name=name,device=device,
									batchSize=batchSize,percGPU=percGPU,
									trainingMethod=trainingMethod,
									decoderActiv=decoderActiv,
									printIter=printIter,dirName=dirName,
									encoderActiv=encoderActiv)

	def __repr__(self):
		return 'NMF\nn_components=%d\nouterIter=%d\nLR=%0.4f\nname=%s\ndevice=%d\nbatchSize=%d\n'%(self.n_components,self.outerIter,self.LR,self.name,self.device,self.batchSize)
	
	def fit(self,x,Winit=None,wg=None):
		'''Fits a previously defined model object
		Inputs
		------
		x : array-like, (n_samples,n_features)
			Data input
		wg : array-like, (n_samples),optional
			Data for weighting 
		Outputs
		-------
		training : dict
			Saves key variables' progression throughout training
			Saves output in tensorflow file
		'''
		x = x.astype(np.float32)
		N,p = x.shape
		self.p = int(p)

		#Adjust the weights
		wg = self._createWG(N,wg)

		#Actually make it random
		Wi = self._createWinit(x,Winit)

		#Set the batch data
		self.I = x.shape[0]
		self.currentBatch = None
		self.batchOrder = None

		#Limit the GPU to self.device
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		dev = str(int(self.device))
		os.environ["CUDA_VISIBLE_DEVICES"] = dev

		#Measures the amount of time to train
		startTime = time.time()

		######################
		# Defining our graph #
		######################
		tf.reset_default_graph()
		
		self._definePlaceholders()
	
		self._defineEncoder()
		self._defineDecoder(Wi)
		self._defineRegularization()
		self._defineMSE()

		#Each model must define its own loss
		self.loss = self.loss_mse# + self.reg_Sd + self.reg_Wd

		#Optimization method
		self._defineOptimization()


		#Initialize all the variables
		self._defineInitialization()

		#Save stuff out
		saver = tf.train.Saver()

		#This gives the cost while we fit the model
		training = {'losses':np.zeros(self.outerIter),
					'mses':np.zeros(self.outerIter)}

		#########################
		# Run our created graph #
		#########################
		#with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
		sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

		if 1 == 1:
			sess.run(self.init)
			
			for kk in range(50000):
				idxs = self._batch()
				xmb = x[idxs]
				wgb = wg[idxs]
				_,er,myWD,mse = sess.run([self.optim_encoder,self.loss,
										self.Wd,self.loss_mse],
										feed_dict={self.x_:xmb,
										self.lr_:self.LR,
										self.wg_:wgb})
				if kk%1000 == 0:
					el = time.time() - startTime
					print('IIteration %d,Time = %0.1f,Loss = %0.5f,MSE = %0.5f'%(int(kk),el,er,mse))


			for i in range(self.outerIter):
				idxs = self._batch()
				xmb = x[idxs]
				wgb = wg[idxs]

				_,er,myWD,mse = sess.run([self.optimstep,self.loss,
										self.Wd,self.loss_mse],
										feed_dict={self.x_:xmb,
										self.lr_:self.LR,
										self.wg_:wgb})

				#Print key variables over time
				if i%1000 == 0:
					el = time.time() - startTime
					print('Iteration %d,Time = %0.1f,Loss = %0.5f,MSE = %0.5f'%(int(i),el,er,mse))

				#Save out loss + reconstruction cost
				training['mses'][i] = mse 
				training['losses'][i] = er 

			#This is tensorflow's saver so we can use the model later
			pn = self.dirName + '/' + self.name + '.ckpt'
			save_path = saver.save(sess,pn)
			self.chkpt = save_path
			self.meta = self.dirName + '/' + self.name + '.ckpt.meta'

			#Save the features
			self.components_ = myWD
		
		return training,sess


