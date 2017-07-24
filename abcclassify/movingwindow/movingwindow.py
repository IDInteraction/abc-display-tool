import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import abcclassify as abcc
import csv
import os.path
from sklearn import tree
import copy


def trainRegion(participant, groundtruth):
    """Apply the groundtruth data to a participant"""
    for index,row in groundtruth.iterrows():
        #print "setting classification for" + str(index) + ":" + str(row["state"])
        participant.setClassification(index, row["state"], testunset = True)
        

def calcWindowedAccuracy(participant, windowsize, advancesize, n_splits = 100):
    """Test the xval accuracy of overlapping windows within the object"""
    firstframe = min(participant.frames)
    lastframe = max(participant.frames)

    windowstarts = range(firstframe, lastframe , advancesize)
    windowends = [x + windowsize for x in windowstarts]
    
    # Set windowends to the smaller of their value or the end of the region
    windowends = [min(x,max(participant.frames)) for x in windowends]
    
       
    means=[]
    stds = [] 
    numclassframes = []
    numframes = []
    
    for (start,end) in zip(windowstarts,windowends):
        
        thispart = participant.split((start, end))
        try: 
            thispartvtc = abcc.videotrackingclassifier(thispart)
            shufsplitscore = thispartvtc.getShuffleSplitScore(n_splits=n_splits)
        except ValueError:
            shufsplitscore = np.array([np.NaN])
        means.append(shufsplitscore.mean())
        stds.append(shufsplitscore.std())
        numclassframes.append(len(thispart.getClassifiedFrames()))
        numframes.append(len(thispart.frames))
        
    
    results = pd.DataFrame.from_items([("startframe", windowstarts),
                                       ("endframe", windowends),
                                  ("mean", means),
                                  ("std", stds),
                                  ("numclassframes", numclassframes),
                                  ("numframes", numframes)])
    
    return results

def plotaccuracy(p1, p2, windowsize, advancesize):
    # Figure modified from
    # https://matplotlib.org/examples/pylab_examples/scatter_hist.html

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_main = [left, bottom, width, height]
    rect_cframes = [left, bottom_h, width, 0.2]


    # start with a rectangular Figure
    plt.figure(1) #, figsize=(8, 8))

    axMain = plt.axes(rect_main)
    axCframes = plt.axes(rect_cframes)
    if min(p1.frames) != min(p2.frames) or         max(p1.frames) != max(p2.frames):
            raise IndexError("Frame ranges must match for both methods")
            
            
    # Calculate results for both methods 
    batchresults = calcWindowedAccuracy(p1, windowsize, advancesize)
    randbatchresults = calcWindowedAccuracy(p2, windowsize, advancesize)
    
    xlimit = (min(p1.frames), max(p1.frames))
    axMain.set_xlim(xlimit)
    axMain.set_ylim((0.6,1.0))
    axCframes.set_xlim(xlimit)
    axMain.plot(batchresults["startframe"], batchresults["mean"], color="blue")
    axMain.plot(randbatchresults["startframe"], randbatchresults["mean"], color="green")

    cframes = list(p1.getClassifiedFrames().index)
    axCframes.scatter(cframes, [1] * len(cframes), s=.1, color="blue", alpha=0.8)
    rcframes = list(p2.getClassifiedFrames().index)
    axCframes.scatter(rcframes, [1.1] * len(rcframes), s=.1, color="green", alpha=0.8)

    plt.show()


