# coding=utf-8
import argparse
import glob
import os
import pickle
import numpy as np

from feature_selection import backwardFeatureSelection,forwardFeatureSelection,getThreshold

#DELTA_GRID = [0,0.0005 , 0.05]
DELTA_GRID = [0.005, 0.01, 0.03, 0.05, 0.1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int) # k-neighbors to compute CMI
    parser.add_argument("--backward", type=str, default="t") # t if backward, otherwise forward FS
    parser.add_argument("--nproc", nargs='?', const=1, type=int) # number of processors
    # parser.add_argument("--ScaleAndNoise", type=str, default="ScaleAndNoise")
    parser.add_argument("--classification", nargs='?', const=0, type=int) # if 0 regression, otherwise classification
    parser.add_argument("--filename", type=str) # full path to file, it must be a pickle
    args = parser.parse_args()

    if args.nproc is None : args.nproc = 1 
    if args.classification is None : args.classification = 0
    # print("Flag : " + str(args.ScaleAndNoise))
    #input_folder = os.path.join(os.getcwd(), "data", "real")
    #filenames = glob.glob(os.path.join(input_folder, "*.pickle")) # get dataset files in pickle format

    # --- DATA LOAD ---
    # for now this tool only supports a single pickle file
    with open(args.filename, 'rb') as fp:
        dataset = pickle.load(fp)

    # --- FEATURE SELECTION ---
    # X: n rows for m features
    # Y: n rows for l targets
    features = dataset["X"] 
    target = dataset["Y"]

    # Scale and Noise if needed
    #features = data_scale(features, args.ScaleAndNoise)
    #target = data_scale(target, args.ScaleAndNoise)

    # grid with different allowed losses w.r.t. the full features dataset
    grid = DELTA_GRID
    grid.sort()

    # if not specified, fixed fraction of samples
    if args.k is None:
        args.k=len(features)//20

    # set the kind of problem and the dictionary that will store the result
    task = 1 if args.classification == 1 else 0 # task=1 -> classification
    
    res = {
        "delta" : [], # list with all deltas
        "numSelected" : [], #
        "selectedFeatures" : [] 
    #    "accuracy" : [] # list of scores associated with the reduced problem
    }
    
    for delta in grid:
        print("Current Delta: {0}".format(delta))
        res["delta"].append(delta)
        threshold = getThreshold(task, target, delta) # for the current delta this is the maximum information we want to loose in backward after pruning the worse features
        print("Current Threshold: {0}".format(threshold))

        if args.backward == 't':
            relevantFeatures = backwardFeatureSelection(threshold, features, target, res, args.k, args.nproc)
        else:
            relevantFeatures = forwardFeatureSelection(threshold,features,target,res,args.k,args.nproc)
            
        res["selectedFeatures"].append(relevantFeatures)
        print("selected Features: {0}".format(res["selectedFeatures"]))
        #res["accuracy"].append(computeAccuracy(task, relevantFeatures, target)) # performance of linear model 
    
    for i in range(len(res["delta"])):
        print("Delta: {0}, final number of features: {1}, selected features IDs: {2}".format(res["delta"][i], res["numSelected"][i], res["selectedFeatures"][i]))
