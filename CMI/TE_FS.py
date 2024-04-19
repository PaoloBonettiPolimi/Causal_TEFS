# coding=utf-8
import numpy as np
from multiprocessing import Pool
from mixedRVMI import CMIEstimate,estimateAllMI

def scoreFeatures(features, target, k, tau, selected=None, flag=0, tau2=None):
    'returns a list of features ID + CMI score on the given target'
    scores = np.zeros(features.shape[1])

    if tau2==None:
        tau2=tau

    for col in range(features.shape[1]):
        if selected is None:
            sel = np.delete(features,col,axis=1)
        else: sel = selected

        candidate = features[tau-1:-1, col].reshape(-1,1) # actual column shifted by 1
        conditioning = sel[tau-1:-1, :] # other columns shifted by 1

        if flag==1:
            conditioning = target[tau-1:-1]
            for lag in range(2,tau+1):
                candidate = np.concatenate((candidate, features[tau-lag:-lag, col].reshape(-1,1)),axis=1)
                if lag<=tau2: conditioning = np.concatenate((conditioning, target[tau-lag:-lag].reshape(-1,1)),axis=1)
        else:
            conditioning = np.concatenate((conditioning, target[tau-1:-1].reshape(-1,1)),axis=1) 
            for lag in range(2,tau+1):
                candidate = np.concatenate((candidate, features[tau-lag:-lag, col].reshape(-1,1)),axis=1)
                conditioning = np.concatenate((conditioning, sel[tau-lag:-lag, :]),axis=1)
                if lag<=tau2: conditioning = np.concatenate((conditioning, target[tau-lag:-lag].reshape(-1,1)),axis=1)

        scores[col] = CMIEstimate(candidate, target[tau:], conditioning, k)
        if scores[col] > 0 : print("CMI: {0}".format(scores[col]))

    return list(zip(range(len(scores)),scores))

def TE_backwardFeatureSelection(threshold,features,target,res,k, nproc, tau, tau2=None):
    'the function returns the selected features starting from the full dataset and removing features keeping the loss of information smaller than the threshold'

    featureScores= []
    relevantFeatures = features # at the beginning all features are included
    idMap = {k: k for k in range(relevantFeatures.shape[1])} # dictionary with original feature position
    CMIScore = 0 # cumulative loss of information
    sortedScores = []

    while CMIScore < threshold and relevantFeatures.shape[1]>1: 
        featureScores = scoreFeatures(features=relevantFeatures, target=target, k=k, tau=tau, tau2=tau2) # for each feature it evaluates the I(Y,X_i|X_A), at first step I(Y,X_i|X_{-i}),...
        
        sortedScores = sorted(featureScores, key=lambda x:x[1]) # ordered list (ascending) based on the score of each feature
        print(sortedScores)
        CMIScore += max(sortedScores[0][1],0) # take the smallest CMI score, if negative consider 0
        if CMIScore >= threshold: break 
        relevantFeatures = np.delete(relevantFeatures, sortedScores[0][0], axis=1) # remove the feature (column) with smallest score
        print("Removing original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
    res["numSelected"].append(relevantFeatures.shape[1])
    return list(idMap.values())

def TE_forwardFeatureSelection(threshold,features,target,res,k, nproc, tau, tau2=None):
    'returns the relevant features for the target starting from an empty array and populating it with the features that have the best CMI score'

    featureScores=[]
    idMap = {k: k for k in range(features.shape[1])} # dictionary with original feature position
    idSelected = []
    selectedFeatures = [np.zeros((features.shape[0]))] # empty array at the beginning
    CMIScore = 0 # cumulative information selected
    n_iters = 0

    remainingFeatures = features # now score the remaining features

    while CMIScore < threshold and n_iters < features.shape[1]:
        if n_iters==0: 
            fl=1
        else: fl=0
        featureScores = scoreFeatures(features=remainingFeatures, target=target, k=k, tau=tau, selected=np.array(selectedFeatures).T, flag=fl, tau2=tau2) 

        sortedScores = sorted(featureScores, key=lambda x:x[1], reverse=True) # scores in descending order
        CMIScore += max(sortedScores[0][1], 0)
        print("Highest CMI score: {0}".format(sortedScores[0][1]))
        if CMIScore >= threshold or sortedScores[0][1] <= 0: break # stop execution also if all scores are negative
        if n_iters==0:
            selectedFeatures = [features[:, idMap[sortedScores[0][0]]]]
        else:
            selectedFeatures.append(features[:, idMap[sortedScores[0][0]]]) # select highest scoring feature
        remainingFeatures = np.delete(remainingFeatures, sortedScores[0][0], axis=1) # best scoring no longer needed for evaluation
        print("Adding original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        idSelected.append(idMap[sortedScores[0][0]])
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
        n_iters += 1
    res["numSelected"].append(np.array(selectedFeatures).T.shape[1]) 
    return idSelected