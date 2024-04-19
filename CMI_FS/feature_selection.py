# coding=utf-8
import numpy as np
from multiprocessing import Pool
from mixedRVMI import CMIEstimate,estimateAllMI

def backwardFeatureSelection(threshold,features,target,res,k, nproc):
    'the function returns the selected features starting from the full dataset and removing features keeping the loss of information smaller than the threshold'

    featureScores= []
    relevantFeatures = features # at the beginning all features are included
    idMap = {k: k for k in range(relevantFeatures.shape[1])} # dictionary with original feature position
    CMIScore = 0 # cumulative loss of information
    sortedScores = []

    while CMIScore < threshold and relevantFeatures.shape[1]>1: 
        if nproc > 1:
            featureScores = scoreParallelFeatures(relevantFeatures, target, k, nproc)
        else: 
            featureScores = scoreFeatures(relevantFeatures, target, k) # for each feature it evaluates the I(Y,X_i|X_A), at first step I(Y,X_i|X_{-i}),...
        
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

def forwardFeatureSelection(threshold,features,target,res,k, nproc):
    'returns the relevant features for the target starting from an empty array and populating it with the features that have the best CMI score'

    featureScores=[]
    idMap = {k: k for k in range(features.shape[1])} # dictionary with original feature position
    idSelected = []
    selectedFeatures = [] # empty array at the beginning
    CMIScore = 0 # cumulative loss of information

    firstBest = sorted(estimateAllMI(features, target, k), key=lambda x:x[1], reverse=True) # ordered list (descending) of features MI scores
    print("----- MI Scores -----")
    print(firstBest)
    print("Best MI score: {0}".format(firstBest[0][1]))
    print("Adding first best original feature: {0}".format(idMap[firstBest[0][0]])) # original feature position
    selectedFeatures.append(features[:, firstBest[0][0]]) # append the best scoring feature to result
    idSelected.append(idMap[firstBest[0][0]]) # save in a list of selected features ID
    
    for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
        if a >= firstBest[0][0]:
            idMap[a] = idMap[a+1]
    idMap.pop(max(idMap))

    remainingFeatures = np.delete(features, firstBest[0][0], axis=1) # now score the remaining features

    while CMIScore < threshold and np.array(selectedFeatures).T.shape[1] < features.shape[1]:
        if nproc > 1:
            featureScores = scoreParallelFeatures(remainingFeatures, target, k, nproc, np.array(selectedFeatures).T)
        else: 
            featureScores = scoreFeatures(remainingFeatures, target, k, np.array(selectedFeatures).T) 

        sortedScores = sorted(featureScores, key=lambda x:x[1], reverse=True) # scores in descending order
        CMIScore += max(sortedScores[0][1], 0)
        print("Highest CMI score: {0}".format(sortedScores[0][1]))
        if CMIScore >= threshold or sortedScores[0][1] <= 0: break # stop execution also if all scores are negative
        selectedFeatures.append(features[:, idMap[sortedScores[0][0]]]) # select highest scoring feature
        remainingFeatures = np.delete(remainingFeatures, sortedScores[0][0], axis=1) # best scoring no longer needed for evaluation
        print("Adding original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        idSelected.append(idMap[sortedScores[0][0]])
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
    res["numSelected"].append(np.array(selectedFeatures).T.shape[1]) 
    return idSelected

def scoreParallelFeatures(features, target, k, nproc, selected=None):
    'parallel version to compute scores'
    args=[]
    for i in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,i,axis=1)
        args.append((features[:, i], target, selected, k))
    with Pool(nproc) as p:
        scores = p.starmap(CMIEstimate, args)
    scores = np.array(scores)
    return list(zip(range(len(scores)),scores))

def scoreFeatures(features, target, k, selected=None):
    'returns a list of features ID + CMI score on the given target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,col,axis=1)
        scores[col] = CMIEstimate(features[:, col], target, selected, k)
        if scores[col] > 0 : print("CMI: {0}".format(scores[col]))

    return list(zip(range(len(scores)),scores))

def getThreshold(task, target, delta):
    if task == 1: # classification task
        return (delta**2)/2
    else:
        return delta/2*np.max(target)**2 # l-infinity norm
