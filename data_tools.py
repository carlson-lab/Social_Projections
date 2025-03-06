'''
LIST OF FUNCTIONS
load_data: Loads and extracts data
getLabels: extracts labels from dictionary
data_subset: extracts subset of data
combine_data: combines datasets
normalize: normalizes data
'''
import json
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

def load_data(filename, fBounds=(1,120), feature_list=['power', 'coherence']):
    """ Loads and extracts data from a JSON file with preprocessed data.
    
    Loads the data from the file, then extracts each field, takes only the data
    within the frequency bounds specified by the input fBounds, and transforms each
    data matrix to be a 2-dimensional matrix where axis 0 is the time window and axis 1
    iterates first by brain area and then by frequency.
    
    INPUTS
    filename: name of the .mat (and .json) file containing the data
        and labels variables. The .mat file should be gzipped, but
        filename should not include .gz. The .json file is not
        gzipped. The .mat file is a list of 3 fields in the order
        power, coherence, granger, and the .json file contains labels.
        LOADED VARIABLES
        power: MxNxP matrix containing the power data for the file,
            where M is frequency, N is brain region, and P is windows.
        coherence: MxNxPxQ matrix containing coherency data for the file,
            where M is frequency, N is window, and P and Q are both
            brain regions.
        labels: Structure containing labeling information for the data. 
            FIELDS:
            'f': dictionary with (k,f) where k is index and f is frequency value.
            'windows': Structure containing information on windows.
                FIELDS:
                'task': Task being done.
                'mouse': Name of mouse.
                'powerFeatures': MxN matrix of string labels describing the
                    features represented in labels.power. M is frequency, N is
                    brain area.
                'cohFeatures': MxPxQ array of string labels describing the
                    features represented in labels.coherence. M is frequency, P
                    and Q are the two brain areas where coherence is calculated.
                'gcFeatures': PxF array of string labels describing the
                    features represented in labels.granger. P iterates over
                    directed pairs of regions, F iterates over frequencies.

    feature_list: list of strings indicating which variables to load from the .mat file
    fBounds: frequency bounds to analyze; default is set to between 1 and 120 Hz, inclusive.
    
    OUTPUTS
    power: Transformed matrix of power values within the frequency range given by fBounds
        in the form MxN, where M is time windows and N is a combination of brain area and
        frequency, iterating first frequency then by area.
    coherence: Transformed matrix of coherency values within the frequency range given by
        fBounds; same dimensions as power. In this case, brain area refers to the pair of brain
        areas being compared.
    granger: Transformed matrix of granger causality values within the frequency range given by
        fBounds; same dimensions as power. In this case, brain area refers to the pair of brain
        areas being compared.
    labels: See labels field of filename above. 
    """

    features = list()
    print(filename)
    with h5py.File(filename, 'r') as file:
        for ft in feature_list:
            features.append(list(file[ft]))

    jfile = filename.replace('.mat','.json')    
    with open(jfile) as f:
        labels = json.load(f)
        
    # only take the data from frequency values within bounds given
    (fLow, fHigh) = fBounds
    fIdx = [k for (k, f) in enumerate(labels['f']) if fLow <= f <= fHigh]

    # convert to array, invert axes, take power, coherency, gc data at
    # indices specified from frequency
    for k,ft in enumerate(feature_list):
        if ft == 'power':
            features[k] = np.asarray(features[k])
            features[k] = np.swapaxes(features[k], 2,0)
            features[k] = features[k][fIdx,:,:]

            # reshape each nd array to matrix after reshaping, axis 0
            # is windows; axis 1 iterates through frequency first,
            # then channel
            a, b, c = features[k].shape
            features[k] = features[k].reshape(a*b, c, order='F').T

            if 'powerFeatures' in labels.keys():
                # reshape corresponding array of feature labels
                # MAKE SURE THESE OPERATIONS CORRESPOND TO OPERATIONS ON ACTUAL FEATURES ABOVE
                pf = np.asarray(labels['powerFeatures'])
                pf = pf[fIdx,:]
                labels['powerFeatures'] = pf.reshape(a*b, order='F')
            
        if ft == 'coherence':
            features[k] = np.asarray(features[k])
            features[k] = np.swapaxes( features[k], 1,2)
            features[k] = np.swapaxes( features[k], 0,3)
            features[k] = np.swapaxes( features[k], 2,3)
            features[k] = features[k][fIdx,:,:,:].astype('float64')
    
            # collect indices of upper triangular portion of brain region x brain region area matrix
            r1, c1 = np.triu_indices( features[k].shape[-1], k=1)
            features[k] = features[k][..., r1,c1]
            
            features[k] = np.swapaxes(features[k],0,1)
            a, b, c = features[k].shape
            features[k] = features[k].reshape(a, b*c, order='F')

            if 'cohFeatures' in labels.keys():
                # reshape corresponding array of feature labels
                # MAKE SURE THESE OPERATIONS CORRESPOND TO OPERATIONS ON ACTUAL FEATURES ABOVE
                cf = np.asarray(labels['cohFeatures'])
                cf = np.swapaxes(cf, 1,2)
                cf = cf[fIdx,:,:]
                cf = cf[:,r1,c1]
                labels['cohFeatures'] = cf.reshape(b*c, order='F')

            
        if ft == 'granger':
            gcFIdx = [k+1 for k in fIdx]

            gcArray = np.asarray(features[k])
            gcArray = gcArray[:, gcFIdx, :]
            features[k] = np.transpose(gcArray, (1,2,0))
            a,b,c = features[k].shape
            features[k] = features[k].reshape(a*b, c, order='F').T

            if 'gcFeatures' in labels.keys():
                # reshape corresponding array of feature labels
                # MAKE SURE THESE OPERATIONS CORRESPOND TO OPERATIONS ON ACTUAL FEATURES ABOVE
                gf = np.asarray(labels['gcFeatures'])
                gf = gf[:, gcFIdx].T
                labels['gcFeatures'] = gf.reshape(a*b, order='F')

            
        if ft == 'xFft':
            labels['s'] = np.asarray(labels['s'])[fIdx]

            W,S = features[k].shape
            features[k] = np.zeros((W,1,S), dtype=np.complex64)
            for w in range(W):
                for s in range(S):
                    a,b = features[k][w,s]
                    features[k][w,0,s] = a + 1j*b
            
    features.append(labels)
            
    return tuple(features)


# def getX(featureArrays, featureWeights=None):
#     '''
#     Prepares the covariate matrix by normalizing and combining all of the 
#     requested features (power, coherence, gc, etc)
    
#     example: X = getX([power, gcFeatures], [1, 0.1])
    
#     INPUT
#     featureArrays: list of feature array names ([power, coherence, gcFeatures])
#     featureWeights: a list of weights by which the corresponding feature will be scaled.    

#     OUTPUT
#     X: normalized, combined feature matrix
#     '''
#     if featureWeights is None:
#         featureWeights = np.ones(len(featureArrays))

#     for feature, w in zip(featureArrays, featureWeights):
#         feature = feature * w/np.std(feature)
#         if 'X' in locals():
#             X = np.concatenate((X, feature),axis=1) # Concatenate so that times line up
#         else:
#             X = feature

#     return X


def get_labels(labels, variable_name = 'task'):
    ''' Generates multivariate labels.
    
    INPUT
    labels: labels variable from preprocessed data
    variableName: name of the field you want for y variables
    
    OUTPUT
    y: task labels for each window '''
    
    task_strings = labels['windows'][variable_name]
    y = LabelEncoder().fit_transform(task_strings)    
    return y 


def data_subset(x, labels, condition):
    """ Returns a subset of the given data.

    INPUTS
    x: numpy array of data (WxF) where W is number of windows and F is number of features
    labels: labels variable from preprocessed data
    condition: boolean list/vector of length W
    """

    x = x[condition]

    lab_copy = deepcopy(labels)
    # iterate over each key in 'windows' dictionary
    for key, value in lab_copy['windows'].items():
        lab_copy['windows'][key] = np.asarray(value)[condition]

    return (x, lab_copy)


def combine_data(x_tup, label_tup):
    """ Combines multiple data subsets together
    Assumes subsets are compatible (i.e. contain same feature space)
    
    INPUTS
    x_tup: tuple of x outputs from data_subset function
    label_tup: tupble of label outputs from data_subset function
    """
    x = np.concatenate(x_tup)
    labels = deepcopy(label_tup[0])

    for key in labels['windows']:
        values = [x for l in label_tup for x in l['windows'][key]]
        labels['windows'][key] = values

    return (x, labels)

