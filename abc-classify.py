#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import argparse
import cv2
import re
from random import shuffle
from sklearn.tree import DecisionTreeClassifier 
from sklearn import preprocessing

# Need video file
# Openface / cppmt data

# Random sample of frames
# Display an image and get prediction


# Construct rpart tree
# Use rpart tree to predict

def loadTrackingData(infile,
        guessClean = True):
    indata = pd.read_csv(infile, index_col = 0)
    stripvals = [x.strip(' ') for x in indata.columns.values]
    indata.columns = stripvals

    if guessClean == True:
        # Remove spurious columns based on filetype
        if set(['timestamp', 'gaze_0_x']).issubset(indata.columns.values):
            print("OpenFace input detected")
            AUCols = [x.find("AU") == 0 for x in indata.columns.values] 
            ControlCols = [True] * 3 + [False] * (indata.columns.values.size - 3)
            mask = [x|y for (x,y) in zip(AUCols, ControlCols)] 

            indata.drop(indata.columns[mask], axis = 1, inplace = True)

    return indata


def getVideoFrame(videosrc, frameNumber):
    # http://stackoverflow.com/questions/11469281/getting-individual-frames-using-cv-cap-prop-pos-frames-in-cvsetcaptureproperty
    fps = videosrc.get(cv2.cv.CV_CAP_PROP_FPS)
    frameTime = 1000 * frameNumber / fps
    videosrc.set(cv2.cv.CV_CAP_PROP_POS_MSEC, frameTime)

    ret, img = videosrc.read()
    if ret == False:
        print "Failed to capture frame" + str(fps)
        quit()

    return img




parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video")

parser.add_argument("--videofile",
        dest = "videofile", type = str, required = True)
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True)
parser.add_argument("--startframe", type = int, required = False)
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False)
parser.add_argument("--minframes", type = int, required = True)

args = parser.parse_args()


cv2.namedWindow("Classification")
videoFile = cv2.VideoCapture(args.videofile)

# TODO test specified frames are within video!
if args.startframe is not None:
    startVideoFrame = args.startframe
else:
    startVideoFrame = videoFile.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

if args.endframe is not None:
    endVideoFrame= args.endframe
else:
    endVideoFrame= videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

print "Tracking between " + str(startVideoFrame) + " and " + str(endVideoFrame)

trackingData = loadTrackingData(args.trackerfile)

# We handle the training period by shuffling all the frames in the video
# We can then work our way through the list as required, to avoid re-drawing the sample
# and risking classifying the same frame twice etc.
trainingframes = range(startVideoFrame, endVideoFrame)
shuffle(trainingframes)

trainedframes = 0

groundtruth = [1,0,0,0,1,1,1,1,0,0]

tree = DecisionTreeClassifier()

# for i in range(-1, args.minframes): 
#     img = getVideoFrame(videoFile, trainingframes[i])
# 
#     cv1.imshow("image", img)
# 
#     key =  cv1.waitKey(0) 
#     groundtruth.append(int(chr(key)))
# 
    
trainedframes = len(groundtruth)


trainingSet = trackingData.loc[trainingframes[0:trainedframes],:]

tree.fit(trainingSet, groundtruth)

print( tree)

predictionSet = trackingData.loc[trainingframes[10:20],:] 

predicted = tree.predict(predictionSet)

print predicted


