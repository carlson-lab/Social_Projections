'''
Creator:
	Austin "The Man" Talbot
Creation Date:
	7/16/2019
Version history
---------------
Version 1.1
Objects
-------
NORM_base
	This contains the base functions necessary to have an NMF model, such 
	as transforming the data, defining the regularization losses and 
	placeholders, saving, and initializing the graph. Not a proper object
	on its own, inheritance to avoid boilerplate code.
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
import numpy.linalg as la
import pickle
import time
from utils import *
from sklearn import decomposition as dp
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

version = "1.0"

class NORM_base(object):
	r'''This is an object containing boilerplate code. Any object that 
		inherits these methods will not need to rewrite them
	Parameters
	----------
	n_components : int
		Number of latent networks we wish to use
	
	outerIter : int
		Number of steps of gradient descent to use in fit
	
	LR : float
		The step size used in gradient descent
	
	name : string
		Name of the model, used for saving
	
	dirName : string
		The directory where we save the model
	
	device : int
		Often a computer will have multiple GPUs and we want to only use
		a subset.
	
	batchSize : int
		Number of points to use for each gradient update
	
	percGPU : float in [0,1]
		The amount of the GPU to use, can run multiple jobs on single GPU
	
	encoderActiv:
		The non-linearity to use for our encoder
	
	Attributes
	----------
	creationDate
		Date and time model created
	
	version
		What version of the model was used. Can be aligned with github
		history.
	
	batchOrder
	I
	currentBatch
		Used by _batch method

	Methods
	-------
	def _batch(self):
		Batches input data for stochastic gradient descent

	def _definePlaceholders(self):
		Defines user-specified inputs for tensorflow

	def _defineEncoder(self):
		Defines our encoder A(X)
	
	def _defineDecoder(self):
		Defines our linear decoder W

	def _defineMSE(self):
		Gets the reconstruction loss ||X-WA(X)||

	def _defineInitialization(self):
		Defines the method that initializes the tensorflow graph

	def transform(self,x):
		Given a trained A(X) we often want to see performance on new data

	def save(self,fileName=None):
		Saves the model as a pickle object

	def save_components(self,fileName=None):
		Saves the components as a CSV file.

	def save_transform(self,X,fileName=None):
		Trasnforms the data and saves to fileName
	
	Examples
	--------
	None
	'''

	def __init__(self,n_components,outerIter=100000,LR=1e-2,k1=128,
					name='aNMF_base',dirName='./tmp',
					device=0,batchSize=100,percGPU=0.33,
					trainingMethod='Adam',decoderActiv='softplus',
					encoderActiv='softplus', printIter=1000):
		self.n_components = int(n_components) 
		self.outerIter= int(outerIter) 
		self.LR = float(LR) 
		self.device = int(device) 
		self.batchSize = int(batchSize) 
		self.k1 = int(k1) 
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(percGPU))
		self.encoderActiv = str(encoderActiv)
		self.decoderActiv = str(decoderActiv) 
		self.name = str(name)
		self.dirName = str(dirName)
		self.printIter = int(printIter)

		self.trainingMethod = str(trainingMethod)

		self.batchOrder = None
		self.I = 0
		self.currentBatch = None

		self.creationDate = dt.now() 
		self.version = version 

	def _batch(self):
		r'''Stochastic gradient descent only calculates the gradient on a 
			subset of the data for each step. This speeds up 
			computation and 
			reduces overfitting. This batch method randomly selects points
			to be used in each step. Done so that all data points are 
			randomly sampled but after a certain number of iterations all
			points have been used. Reduces variance compared to true 
			stochastic.
		Parameters
		----------
		None
		Returns
		-------
		idxs : bool-array, [self.batchSize]
			Boolean array of points to used for each iteration
		'''
		#Set things the first time
		if self.currentBatch is None:
			self.currentBatch = self.I

		batchMax = self.currentBatch*self.batchSize

		#Remake the permutation
		if batchMax >= self.I:
			self.currentBatch = 0 
			self.batchOrder = rand.permutation(np.arange(self.I))

		batchStart = self.currentBatch*self.batchSize
		batchEnd = (self.currentBatch+1)*self.batchSize
		idxs = np.zeros(self.batchSize)

		if batchEnd > self.I:
			#We go off the end
			n_include = self.I - batchStart
			n_random = self.batchSize - n_include

			#Include the last few
			idxs[:n_include] = self.batchOrder[batchStart:]
			#Randomly select from all data
			idxs[n_include:] = rand.randint(0,self.I,size=n_random)
		else:
			#We don't go off the end
			idxs = self.batchOrder[batchStart:batchEnd]
		
		self.currentBatch += 1
		idxs = idxs.astype(np.int_)

		return idxs

	def _definePlaceholders(self):
		r'''This defines all the data placeholders needed for the 
			NMF models. These are data input by the user during training. 
			Not all placeholders are used in all models
		Parameters
		----------
		None
		Returns
		-------
		None
		Sets
		----
		x_ : tf.placeholder, [self.batch_size,self.p]
			This is the placeholder for our features, in the LPNE lab these
			are generally power and coherence.
		y_ : tf.placeholder, [self.batch_size]
			This is the placeholder for our prediciton. Not used in NMF
		wg_ : tf.placeholder, [self.batch_size,1]
			This is the placeholder for the generative weights. Used to 
			make sure the model doesn't model one group more effectively due
			to unbalanced data
		ws_ : tf.placeholder, [self.batch_size,1]
			This is the placeholder for the supervision weight. Used to 
			deal with unbalanced classes. Not used in NMF
		wa_ : tf.placeholder, [self.batch_size,1]
			Adversary weights Not used in NMF
		lr_ : tf.placeholder, []
			This is the learning rate for optimizaiton. Can be used for 
			annealing to ensure convergence with stochastic algorithm
		mu_ : tf.placeholder, []
			This is the supervision strength. Can be used to anneal 
			predictability
		'''
		bs = self.batchSize
		self.x_ = tf.placeholder(dtype=tf.float32,
											shape=[None,self.p],name='x_')
		self.y_ = tf.placeholder(dtype=tf.float32,shape=[None],name='y_')
		self.wg_ = tf.placeholder(dtype=tf.float32,shape=[None,1],name='wg_')
		self.ws_ = tf.placeholder(dtype=tf.float32,shape=[None,1],name='ws_')
		self.ws2_ = tf.placeholder(dtype=tf.float32,shape=[None,1],
															name='ws2_')
		self.wa_ = tf.placeholder(dtype=tf.float32,shape=[None,1],name='wa_')
		self.mu_ = tf.placeholder(dtype=tf.float32,shape=[],name='mu_')
		self.lr_ = tf.placeholder(dtype=tf.float32,shape=[],name='lr_')


	def _defineEncoder(self,Winit):
		r'''This defines ethe encoder A(X) and saves it as an object 
			attribute.
		Parameters
		----------
		None 
		Returns
		-------
		None
		Sets
		----
		austin1, austin3, ...
			These are the encoding networks that map X to the scores A(X)
		sd, tf.array, [self.batch_size,self.n_components]
			These are the oficial scores rectified to be non-negative using
			the softplus function
		'''
		if self.encoderActiv == 'softplus':
			activ = tf.nn.softplus
		elif self.encoderActiv == 'sigmoid':
			activ = tf.nn.sigmoid
		else:
			print(self.encoderActiv)
			print('Unrecognized encoder activation')

		output = tf.nn.softplus
		Ainit = Winit.dot(la.inv(np.dot(Winit.T,Winit))).astype(np.float32)
		Binit = np.zeros(self.n_components).astype(np.float32)
		encoderStyle = 'singleLayer'

		if encoderStyle == 'singleLayer':
			with tf.variable_scope('encoder'):
				#This encodes the data into a latent representation
				self.A_ = tf.Variable(Ainit,name='A_')
				self.B_ = tf.Variable(Binit,name='B_')
				self.sdrm = tf.matmul(self.x_,self.A_,name='sdrm')
				self.sdr = tf.add(self.sdrm,self.B_,name='sdr')
				self.sd = output(self.sdr,name='sd')
		else:
			print(encoderStyle)
			print('Unrecognized encoder')
	
	def _createWinit(self,X,Winit):
		if Winit is None:
			model = dp.NMF(self.n_components, init='random')
			model.fit(X)
			comp = model.components_
			WinitT = comp.T
			power = (WinitT**2).mean(axis=0)
			WinitT2 = WinitT/power
			Winit = WinitT2.astype(np.float32)
		else:
			Winit = Winit.astype(np.float32)

		S = model.fit_transform(X)
		print('<<<<<<<<<<<<<')
		print(np.mean(S))

		return Winit

	def _createSWinit(self,X,Y,Winit):
		if Winit is None:
			model = dp.NMF(self.n_components, init='random')
			S_train = model.fit_transform(X)
			model_lr = LogReg()
			model_lr.fit(S_train,Y)

			# sort components by initial 'predictiveness'
			coeffs = np.abs(model_lr.coef_)
			idxs = np.argsort(coeffs)[0]
			comps = model.components_
			Winit = comps[idxs]
			
			Winit = Winit.astype(np.float32)
		else:
			Winit = Winit.astype(np.float32)

		return Winit.T
	
	def _createWS(self,N,ws):
		if ws is None:
			ws = np.ones((N,1)).astype(np.float32)
		else:
			ws = ws/np.mean(ws)
			ws = ws.astype(np.float32)
		return ws
	
	def _createWG(self,N,wg):
		if wg is None:
			wg = np.ones((N,1)).astype(np.float32)
		else:
			wg = wg/np.mean(wg)
			wg = wg.astype(np.float32)
		return wg

	def _defineDecoder(self,Winit):
		r'''This defines our decoder, which for NMF is linear map W
		Parameters
		----------
		None 
		Returns
		-------
		None
		Sets
		----
		Wdl : tf.Variable, [self.n_components,self.p]
			The log of our linear mapping. Allows for unconstrained opt
		Wd : tf.matrix, [self.n_components,self.p]
			This is our linear mapping rectified using softplus
		xd : tf.matrix, [self.batchSize,self.p]
			This is the reconstructed data WA(x_)
		'''
		with tf.variable_scope('decoder'):
			self.decoderActiv = 'softplus'
			#This reconstructs data given latent representation
			#Wdl p x n_components
			if self.decoderActiv == 'relu':
				self.Wdl = tf.Variable(Winit.astype(np.float32),name='Wdl')
				self.Wdr = tf.nn.relu(self.Wdl,name='Wdr')
			elif self.decoderActiv == 'softplus':
				Wi2 = np.log(np.exp(Winit)-1+.001)
				self.Wdl = tf.Variable(Wi2.astype(np.float32),name='Wdl')
				self.Wdr = tf.nn.softplus(self.Wdl,name='Wdr')
			else:
				print('Unrecognized decoder activation')
			squared = tf.square(self.Wdr)
			power = tf.reduce_mean(squared,axis=0)
			div = tf.sqrt(power)
			self.Wdt = tf.divide(self.Wdr,div,name='Wdt')
			self.Wd = tf.transpose(self.Wdt,name='Wd')

			self.xd = tf.matmul(self.sd,self.Wd,name='xd')

	def _defineMSE(self):
		r'''This defines the reconstruction loss, mean squared error
		Parameters
		----------
		None
		Returns
		-------
		None
		Sets
		----
		loss_mse : tf.float
			The reconstruction cost ||X - WA(X)||^2
		'''
		diff = self.x_ - self.xd
		self.diff2 = tf.square(diff,name='diff2') 
		self.myMSE = tf.reduce_mean(self.diff2,axis=1,name='myMSE')
		self.wMSE = tf.multiply(self.myMSE,tf.squeeze(self.wg_))
		self.loss_mse = tf.reduce_mean(self.wMSE)
	
	def _defineOptimization(self):
		r'''Define the optimizer

		'''
		if self.trainingMethod == "GradientDescent":
			self.optimstep = tf.train.GradientDescentOptimizer(learning_rate=self.lr_).minimize(self.loss)
			self.optim_encoder = tf.train.GradientDescentOptimizer(learning_rate=self.lr_).minimize(self.loss,var_list=variables_from_scope('encoder'))
		elif self.trainingMethod == "Momentum":
			self.optimstep = tf.train.MomentumOptimizer(learning_rate=self.lr_).minimize(self.loss)
			self.optim_encoder = tf.train.MomentumOptimizer(learning_rate=self.lr_).minimize(self.loss,var_list=variables_from_scope('encoder'))
		elif self.trainingMethod == "Adam":
			self.optimstep = tf.train.AdamOptimizer(learning_rate=self.lr_).minimize(self.loss)
			self.optim_encoder = tf.train.AdamOptimizer(learning_rate=self.lr_).minimize(self.loss,var_list=variables_from_scope('encoder'))
		else:
			print('Unrecognized training method')

	def _defineInitialization(self):
		r'''In tensorflow there is a deference between defining a graph and
			initializing it. This initializes the graph
		Parameters
		----------
		None
		Returns
		-------
		None
		Sets
		----
		init
			Initializes the graph
		'''
		init_global = tf.global_variables_initializer()
		init_local = tf.local_variables_initializer()
		self.init = tf.group(init_global,init_local)

	def transform(self,x):
		'''This method computes the network strengths at each time interval
		Inputs:
		x : array-like, (n_samples,n_features)
			Data input
		--------
		Outputs:
		s : array-like, (n_samples,n_components)
			Scores for transformed data
		'''
		x = x.astype(np.float32)
		I = x.shape[0]
		bs = self.batchSize

		#Limit the GPU to self.device
		#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		#dev = str(int(self.device))
		#os.environ["CUDA_VISIBLE_DEVICES"] = dev

		
		checkpoint = tf.train.latest_checkpoint(self.dirName)
		with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
			new_saver = tf.train.import_meta_graph(self.meta)

			graph = tf.get_default_graph()
			new_saver.restore(sess,checkpoint)

			x_ = graph.get_tensor_by_name('x_:0')
			scores = graph.get_tensor_by_name('encoder/sd:0')
			mySd = sess.run(scores,feed_dict={x_:x},options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

		return mySd
	
	def save_components(self,fileName=None):
		r'''This saves the components of our mode, the linear mapping
		Parameters
		----------
		fileName : string,optional
			This saves 
		Returns
		-------
		None
		Example
		-------
		W_true = np.abs(rand.randn(30,1000))
		S_true = np.abs(rand.randn(100000,30))
		X = np.dot(S_true,W_true) + rand.randn(100000,1000)
		model = NMF(30)
		model.fit(X)
		model.save_components('TestFile.csv')
		'''
		if fileName is None:
			fName = self.name + '.csv'
		else:
			fName = fileName

		np.savetxt(fName,self.components_,fmt='%0.8f',delimiter=',')
	
	def save_transform(self,X,fileName=None):
		r'''This transforms the data and saves as csv. This saves all
		networks.
		Parameters
		----------
		fileName : string,optional
			This saves 
		Returns
		-------
		None
		Example
		-------
		W_true = np.abs(rand.randn(30,1000))
		S_true = np.abs(rand.randn(100000,30))
		X = np.dot(S_true,W_true) + rand.randn(100000,1000)
		model = NMF(30)
		model.fit(X)
		model.save_components('TestFile.csv')
		'''
		S = self.transform(X)
		if fileName is None:
			fName = self.name + '_scores.csv'
		else:
			fName = fileName

		np.savetxt(fName,S,fmt='%0.8f',delimiter=',')

	def save(self,fileName=None):
		r'''Saves the model as a pickle object so it can be used later
		Parameters
		----------
		fileName : string,optional
			The model is saved as a pickle
		Returns
		-------
		Nones
		References
		----------
		https://www.youtube.com/watch?v=NQuryN_P0l8
		'''
		if fileName is None:
			fileName = self.name + '.p'
		myDict = {'model':self}
		pickle.dump(myDict,open(fileName,'wb'))
	
	def getUKU(self):
		components = self.components_
		power = components**2
		sumPower = np.sum(power,axis=0)
		UKU = components/sumPower
		return UKU
	
	def plotCSFA(self,C,nFreq,l,regionList=[],saveName='Default.png'):
		UKU = self.getUKU()
		power_features = UKU[l,:C*nFreq]
		fig,axs = plt.subplots(C,C)
		freqs = np.arange(1,nFreq+1)
		
		#This plots the powers
		for i in range(C):
			axs[i,i].plot(freqs,power_features[i*nFreq:(i+1)*nFreq])
			if len(regionList) > 0:
				axs[i,i].set_title(regionList[i])

		#Now lets plot the coherences
		coh_features = UKU[l,C*nFreq:]
		count = 0 
		for i in range(C-1):
			for j in range(i,C):
				feats = coh_features[count*nFreq:(count+1)*nFreq]
				axs[i,j].plot(freqs,feats)
				count += 1

		fig.savefig(saveName)


def train_model(data, options, transform_batch=1000):
	X,y = data

	# set options to defaults if not included in dictionary
	nC = options.get('n components', 10)
	LR = options.get('learning rate', 1e-4)
	mu = options.get('superstrength', 3)
	outerIter = options.get('n iter', 400000)
	device = options.get('gpu', 0)
	name = options.get('name', datetime.now().strftime("nmfmodel_%Y_%m_%d_%H:%M:%S:%f"))
	dirName = options.get('dir name', name + 'Dir')
	n_blessed = options.get('n supervised', 1)
	samp_weights = options.get('sample weights', np.ones_like(y, dtype=np.float32))
	perc_gpu = options.get('gpu usage', 0.95)
	repeats = options.get('repeats', 20)
	metric = options.get('train metric', 'auc')

	# save list of models trained with different random intializations
	model_nmf = list()
	model_lr = list()
	perf = np.zeros(repeats)
	for r in range(repeats):

		model_nmf.append(options['NMF variant'](nC, LR=LR, mu=mu, outerIter=outerIter,
						   device=device, dirName=dirName,
						   n_blessed=n_blessed, name=name, percGPU=perc_gpu))
		model_nmf[r].fit(X, y, ws=samp_weights[:,np.newaxis], wg=samp_weights[:,np.newaxis], return_flag=False)

		# caclculate scores in batches
		k = 0
		W = X.shape[0]
		S = np.empty((W, nC))
		while k < W:
			k_next = k+transform_batch
			Xb = X[k : k_next] 
			S[k : k_next] = model_nmf[r].transform(Xb)
			k = k_next

		model_lr.append(LogReg('l1'))
		model_lr[r].fit(S[:,:model_nmf[r].n_blessed], y, sample_weight=samp_weights)
		
		perf[r], _ = evaluate_model((model_nmf[r], model_lr[r]), (X,y), metric)

	best_idx = np.argmax(perf)
	model = (model_nmf[best_idx], model_lr[best_idx])
	return model


def evaluate_model(model, data, metric):
	X,y = data
	model_nmf, model_lr = model
		
	S = model_nmf.transform(X)
	S_blessed = S[:,:model_nmf.n_blessed]

	perf_data = dict()
	perf_data['scores'] = S_blessed
	if metric is 'accuracy':
		performance = model_lr.score(S_blessed, y)

	else:
		perf_data['d_func'] = model_lr.decision_function(S_blessed)
		lb = LabelBinarizer()
		y_1hot = lb.fit_transform(y)

		if metric =='auc':
			perf_data['ovr_auc'] = roc_auc_score(y_1hot, perf_data['d_func'], average=None)
			performance = np.mean(perf_data['ovr_auc'])
		elif metric == 'precision':
			perf_data['ovr_avg_prec'] = average_precision_score(y_1hot, perf_data['d_func'],
										average=None)
			performance = np.mean(perf_data['ovr_avg_prec'])

	return performance, perf_data
