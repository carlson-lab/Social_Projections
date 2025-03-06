'''
Creator:
	Austin "The Man" Talbot
Creation Date:
	7/16/2018
Version history
---------------
Version 1.0-1.2
	
Objects
-------
sNMF
	This allows for basic logistic supervision for a subset of features. 
	The number of blessed features is user specified.
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

class sNMF(NORM_base):
	r'''An encoded non-negative matrix factorization with supervision

	Optimizes the following problem

	min_{A,W} ||X-WA(X)||^2 + ||Y-DA(X)||^2 + reg*||A(X)||^2 + reg*||W||^2

	Inherits from
	-------------
	aNMF_base

	Attributes added
	----------------
	mu : float
		Supervision strength

	n_blessed : int
		Number of supervised networks

	Methods added
	-------------
	def fit(self,x,wg=None):
		X is our data of interest and wg our our weights for importance
		in reconstruction

	References
	----------
	Future JASA paper (hopefully)
	'''
	def __init__(self,n_components,outerIter=100000,
				LR=1e-4,name='Default_sNMF',device=0,batchSize=100,
				n_blessed=3,mu=.8,dirName='./stmp',percGPU=0.49,
				trainingMethod='Adam',printIter=1000,
				encoderActiv='softplus',decoderActiv='softplus'):
		super(sNMF,self).__init__(n_components,outerIter=outerIter,
									LR=LR,name=name,
									dirName=dirName,device=device,
									batchSize=batchSize,percGPU=percGPU,
									trainingMethod=trainingMethod,
									decoderActiv=decoderActiv,
									printIter=printIter,
									encoderActiv=encoderActiv)
		self.n_blessed = int(n_blessed)
		self.mu = float(mu)
	
	def __repr__(self):
		return 'sNMF\nn_components=%d\niters=%d\nLR=%0.5f\nname=%s\ndevice=%d\nversion=%s\nbatchSize=%d\nn_blessed=%d\nmu=%0.4f\ndirName=%s'%(self.n_components,self.outerIter,self.LR,self.name,self.device,self.version,self.batchSize,self.n_blessed,self.mu,self.dirName)

	
	def fit(self,x,y,Winit=None,ws=None,wg=None,return_flag=True):
		x = x.astype(np.float32)
		N,p = x.shape
		y = y.astype(np.float32)
		self.p = int(p)

		#Adjust the weight
		ws = self._createWS(N,ws)
		wg = self._createWG(N,wg)
		Wi = self._createSWinit(x,y,Winit)

		#Set the batch data
		self.I = x.shape[0]
		self.currentBatch = None
		bs = self.batchSize
		self.batchOrder = None

		#Limit the GPU to self.device
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		dev = str(int(self.device))
		os.environ["CUDA_VISIBLE_DEVICES"] = dev

		#Timing function
		startTime = time.time()

		######################
		# Defining our graph #
		######################
		tf.reset_default_graph()

		self._definePlaceholders()

		self._defineEncoder(Wi)
		self._defineDecoder(Wi)

		with tf.variable_scope('decoder'):
			self.Phi = tf.get_variable('Phi',shape=[self.n_blessed,1])
			self.B_ = tf.get_variable('B_',shape=[1])
		self.yd = tf.squeeze(tf.matmul(self.sd[:,:self.n_blessed],self.Phi)) + self.B_

		#Define our reconstruction loss
		self._defineMSE()

		#Supervision loss
		self.ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_,
												logits=self.yd)
		self.wce = tf.multiply(tf.squeeze(self.ws_),self.ce)
		self.loss_sup = tf.reduce_mean(tf.square(self.wce))

		#Total loss
		self.loss = self.loss_mse + tf.scalar_mul(self.mu_,self.loss_sup)

		#Optimization step
		self._defineOptimization()

		#Initialize all the variables
		self._defineInitialization()

		#Save stuff out
		saver = tf.train.Saver()

		#Save different values while we fit model
		training = {}
		training['losses'] = np.zeros(self.outerIter)
		training['mses'] = np.zeros(self.outerIter)
		training['sups'] = np.zeros(self.outerIter)
		training['sdSup'] = np.zeros(self.outerIter)

		sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

		if 1 == 1:
			sess.run(self.init)

			mu = 0.001
			for kk in range(100):
				idxs = self._batch()
				xmb = x[idxs]
				ymb = np.squeeze(y[idxs])
				wgmb = wg[idxs]
				wsmb = ws[idxs]
				_,er,myWD,mse = sess.run([self.optim_encoder,self.loss,
											self.Wd,self.loss_mse],
											feed_dict={self.x_:xmb,
											self.y_:ymb,
											self.lr_:self.LR,
											self.ws_:wsmb,self.mu_:mu,
											self.wg_:wgmb})
				if kk%1000 == 0:
					el = time.time() - startTime
					print('Iteration %d,Time = %0.1f,Loss = %0.5f,MSE = %0.5f'%(int(kk),el,er,mse))
					Ar = sess.run(self.A_)
					mySD,myWD = sess.run([self.sd,self.Wd],
											feed_dict={self.x_:xmb,
											self.y_:ymb,
											self.lr_:self.LR,
											self.ws_:wsmb,self.mu_:mu,
											self.wg_:wgmb})
					print(np.mean(Ar),np.mean(mySD),np.mean(myWD))

			for i in range(self.outerIter):
				#Anneal our supervision strength
				mu = np.minimum(1.00005*mu,self.mu)

				#Get our batch variables
				idxs = self._batch()
				xmb = x[idxs]
				ymb = np.squeeze(y[idxs])
				wgmb = wg[idxs]
				wsmb = ws[idxs]

				_,error,mySD,myWD,mse,sup,ph = sess.run([self.optimstep,
									self.loss,self.sd,self.Wd,
									self.loss_mse,self.loss_sup,
									self.Phi],feed_dict={self.x_:xmb,
									self.y_:ymb,self.wg_:wgmb,
									self.ws_:wsmb,self.mu_:mu,
									self.lr_:self.LR})

				training['losses'][i] = error
				training['mses'][i] = mse
				training['sups'][i] = sup
				training['sdSup'][i] = np.mean(mySD[:,:self.n_blessed])

				if i%1000 == 0:
					el = time.time() - startTime
					print('Iteration %d,Time = %0.1f,MSE = %0.5f,SUP = %0.5f,SD_SUP = %0.3f,'%(int(i),el,mse,sup,np.mean(mySD[:,:self.n_blessed])))
					print(np.mean(mySD))
					print(sess.run(self.Phi))
					print(mu)

			pn = self.dirName + '/' + self.name + '.ckpt'
			save_path = saver.save(sess,pn)
			self.chkpt = save_path
			self.meta = self.dirName + '/' + self.name + '.ckpt.meta'

			#Parameters
			self.phi = ph
			self.components_ = myWD

			if return_flag:
				like = sess.run(self.loss_mse,feed_dict={self.x_:x,
									 self.mu_:mu,self.wg_:wg,self.ws_:ws})
				yy = np.squeeze(y)
				dlike = sess.run(self.loss_sup,feed_dict={self.x_:x,self.y_:yy,
									  self.mu_:mu,self.wg_:wg,self.ws_:ws})


				return training,sess,like,dlike
			else:
				return training,sess


