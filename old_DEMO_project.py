import numpy as np
import sklearn.decomposition as dp
import pickle
import sys,os
import numpy.random as rand
from norm_encoded import NMF 
from norm_supervised import sNMF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import auc,roc_curve
from sklearn.utils.random import sample_without_replacement
import scipy.io
from data_tools import load_data


#################
##             ##
##  How to run ##
##             ##
#################
#python DEMO_project.py data_file.mat save_file.mat


########################################################
# This is the name of the .mat file to store your data #
########################################################
fileName = str(sys.argv[1])

#Actually load in the data using the correct contemporary loader
myList = load_data(fileName,fBounds=(1,56),
						feature_list=['power','coherence','granger'])

#Gets the power, coherence and exp(granger) features along with truncation
power = myList[0]
coherence = myList[1]
granger = myList[2]
granger = np.exp(granger)
granger[granger>10] = 10

# Load the labels that matter, the time stamps (time)
# the mouse identity (mouse)
# experiment date (expDate)
# and window number (windows)
labels = myList[3]
windows = labels['windows']
time = windows['Time']
mouse = windows['mouse']
expDate = windows['expDate']

#We downweight the importance of coherence and granger by 10
print('New weighting scheme')
powerS = power
coherenceS = .1*coherence
grangerS = .1*granger

#Combine into data matrix
X = np.hstack((powerS,coherenceS,grangerS))
X = 10.0*X


#This code creates an object with the proper number of components
#This is super ugly because TF 1.0 is terrible
nIter=5
dev=3
number_test = str(3906)
number_components = str(9)
trial = str(6)
dirName = './supervised_rep_' + number_test + number_components + trial
supStr=3.0
model = sNMF(5,outerIter=nIter,device=dev,dirName=dirName,LR=1e-5,
                percGPU=.35,
                n_blessed=1,mu=supStr)

#Actually load mapping from subdirectory
model.meta = model.dirName + '/' + model.name + '.ckpt.meta'

#Transform coordinates
S_DEP = model.transform(X)


#Save relevant variables in a dictionary that will be saved as a .mat file
myDict = {'Scores_supervised':S_DEP[:,0],
			'Scores_unsupervised':S_DEP[:,1:],
			'mouse':mouse,'expDate':expDate,'time':time}

#This is the name of the file you wish to save to
saveName = sys.argv[2]
scipy.io.savemat(saveName,myDict)



