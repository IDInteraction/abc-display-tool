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
from sklearn import metrics

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

def loadExternalGroundTruth(infile, format = "checkfile"):
    if format != "checkfile":
        print "Only checkfiles implemented"
        sys.exit()

    indata = pd.read_csv(infile, index_col=0,
            names = ["frame", "x", "y", "w", "h", "state"])

    indata.drop(["x","y","w","h"], axis=1 ,inplace = True)

    return indata


def getVideoFrame(videosrc, frameNumber):
    # Return the video frame from the video
    # Pass in a **1 INDEXED** video frame number
    # http://stackoverflow.com/questions/11469281/getting-individual-frames-using-cv-cap-prop-pos-frames-in-cvsetcaptureproperty
    fps = videosrc.get(cv2.cv.CV_CAP_PROP_FPS)
    frameTime = 1000 * (frameNumber-1) / fps
    videosrc.set(cv2.cv.CV_CAP_PROP_POS_MSEC, frameTime)

    ret, img = videosrc.read()
    if ret == False:
        print "Failed to capture frame" + str(frameNumber - 1)
        sys.exit()

    return img

def runClassifier(traininggroundtruth, trainingtrackingdata,
        evaluationgroundtruth, evaluationtrackingdata):


    tree = DecisionTreeClassifier()

    trainedframescount  = len(traininggroundtruth)
    if len(trainingtrackingdata.index) != trainedframescount:
        print "Size mismatch"
        sys.exit()
    print "Classifying with " + str(trainedframescount) + " frames" 
    print "Evaluating with  " + str(len(evaluationgroundtruth)) + " frames"

    tree.fit(trainingtrackingdata, traininggroundtruth)

    predicted = tree.predict(evaluationtrackingdata)
    
    print(metrics.classification_report(evaluationgroundtruth, predicted))
    print(metrics.confusion_matrix(evaluationgroundtruth, predicted))
    print(metrics.accuracy_score(evaluationgroundtruth, predicted))
    






parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video")

parser.add_argument("--videofile",
        dest = "videofile", type = str, required = True)
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True)
parser.add_argument("--startframe", type = int, required = False)
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False)
#parser.add_argument("--minframes", 
#        dest = "minframes", type = int, required = True)
parser.add_argument("--extgt", type = str, required = False)
parser.add_argument("--entergt",
        dest = "entergt", action="store_true")
parser.add_argument("--useexternalgt", 
        dest = "entergt", action='store_false')
parser.add_argument("--externaltrainingframes", type = int, required = False)

parser.set_defaults(entergt=True)

args = parser.parse_args()

if (not args.entergt) and args.extgt is None:
    print "If not entering ground-truth from video frames, external ground truth must be provided"
    sys.exit()

cv2.namedWindow("Classification")
videoFile = cv2.VideoCapture(args.videofile)

if args.startframe is not None and args.endframe is not None:
    if args.startframe < 1:
        print "Startframe must be >=1"
        sys.exit()
    if args.endframe < 1:
        print "Endframe muse be >=1"
    if args.endframe <= args.startframe:
        print "Startframe must be before endframe"
        sys.exit()


if args.startframe is not None:
    startVideoFrame = args.startframe
else:
    startVideoFrame = videoFile.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

if args.endframe is not None:
    lastframe = videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    if lastframe < args.endframe:
        print "Endframe is after the end of the video"
        sys.exit()
    endVideoFrame =  args.endframe
else:
    endVideoFrame= videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

print "Tracking between " + str(startVideoFrame) + " and " + str(endVideoFrame)

trackingData = loadTrackingData(args.trackerfile)

# We handle the training period by shuffling all the frames in the video
# We can then work our way through the list as required, to avoid re-drawing the sample
# and risking classifying the same frame twice etc.
trainingframes = range(startVideoFrame, endVideoFrame)
shuffle(trainingframes)

if args.extgt is not None:
    print "Loading external ground-truth file"
    externalGT = loadExternalGroundTruth(args.extgt)
    

trainedframescount = 0


groundtruth = []

if args.entergt:
    while trainedframescount < len(trainingframes):
        thisframe = trainingframes[trainedframescount]
        img = getVideoFrame(videoFile,thisframe) 
 
        cv2.imshow("Classification", img)
 
        key =  cv2.waitKey(0) 
        if(chr(key) == 'c'):
            runClassifier(groundtruth[:(trainedframescount/2)],
                    trackingData.loc[trainingframes[:(trainedframescount/2)]],
                    groundtruth[(trainedframescount/2):],
                    trackingData.loc[trainingframes[(trainedframescount/2):trainedframescount]])
            if args.extgt is not None:
                print "Classification using all remaining external ground truth data:"
                runClassifier(groundtruth[:(trainedframescount/2)],
                        trackingData.loc[trainingframes[:(trainedframescount/2)]],
                        externalGT.loc[trainingframes[trainedframescount:],"state"],
                        trackingData.loc[trainingframes[trainedframescount:]])
        else:
            groundtruth.append(int(chr(key)))
            trainedframescount = trainedframescount + 1
            if args.extgt is not None:
                print int(externalGT.loc[thisframe])
        print str(trainedframescount) + " frames classified"

else:
    if args.externaltrainingframes is not None:
        trainedframescount = args.externaltrainingframes
    else:
        trainedframescount = int(raw_input("Enter training frames: "))
    groundtruthDF = externalGT.loc[trainingframes[:trainedframescount],"state"]
    groundtruth = list(groundtruthDF)

    runClassifier(groundtruth[:(trainedframescount/2)],
                  trackingData.loc[trainingframes[:(trainedframescount/2)]],
                  groundtruth[(trainedframescount/2):],
                  trackingData.loc[trainingframes[(trainedframescount/2):trainedframescount]])

    print "Classification using all remaining external ground truth data:"
    runClassifier(groundtruth[:(trainedframescount/2)],
                        trackingData.loc[trainingframes[:(trainedframescount/2)]],
                        externalGT.loc[trainingframes[trainedframescount:],"state"],
                        trackingData.loc[trainingframes[trainedframescount:]])

