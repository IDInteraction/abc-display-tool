#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import argparse
import cv2
import re
from sklearn.tree import DecisionTreeClassifier 
from sklearn import preprocessing
from sklearn import metrics
import pickle

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

    tree = DecisionTreeClassifier() #random_state = rndstate)

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
    
    return tree


def savePredictions(inputtree,  alltrackingdata, evaluationframes, filename):
    # Save the predictions from a tree
    predictiontrackingdata = alltrackingdata.loc[evaluationframes]
    predicted = inputtree.predict(predictiontrackingdata)

    predframe = pd.DataFrame(
            {'frame' : evaluationframes, 'attention' : predicted}
            )
    predframe.sort_values("frame",inplace = True)
    predframe.to_csv(filename, index=False, columns=[ "frame","attention"])



##############################################




parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video.  For each frame enter a numeric behaviour state.  Press c to classify based on half of the classified frames.  The remaining frames are used to evaluate the performance of the classifier.  Can optionally use an external ground truth file for classification and/or verification.")

parser.add_argument("--videofile",
        dest = "videofile", type = str, required = True,
        help = "The input video file to classify")
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True,
        help = "The data from some object tracking software.  Currently only OpenFace data are supported")
parser.add_argument("--startframe", type = int, required = False,
        help = "The frame of the video to start classifcation at.  Defaults to start of video")
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False,
        help = "The end frame to run classification on.  Defaults to the end of the video")
parser.add_argument("--extgt", type = str, required = False,
        help = "Whether to use an external ground truth file. (currently) assumed to have 6 columns; the first containing the video frame number, the sixth containing the state" )
parser.add_argument("--entergt",
        dest = "entergt", action="store_true",
        help = "Whether to interactively enter ground truth data.  For each frame enter a numeric state, c to classify or u to undo the previous frame")

parser.add_argument("--useexternalgt", 
        dest = "entergt", action='store_false',
        help = "Whether to use the externally specified ground truth file for classification, instead of classifying interactivel")
parser.add_argument("--externaltrainingframes", type = int, required = False,
        help = "The number of frames to use for training and local classification if using an external ground truth file.  Will be prompted for if not specified")
parser.set_defaults(entergt=True)

parser.add_argument("--shuffle", dest="shuffle", action="store_true",
        help = "Whether to classify frames in a random order (default)")
parser.add_argument("--noshuffle", dest="shuffle", action="store_false",
        help = "Whether to classify frames in the order they appear in the video")
parser.set_defaults(shuffle=True)

parser.add_argument("--outfilelocalpreds",
        dest="outfilelocalpreds", type = str, required = False,
        help = "The filename to output classifier performance on the locally specified data to - i.e. data that have been classified interactively (or via the externaltrainingframes argument)")
parser.add_argument("--outfileexternalpreds",
        dest="outfileexternalpreds", type = str, required = False,
        help = "The filename to output classifier performance on the externally specified data to - i.e. data in the external groundtruth file, that haven't been used for training or local classifier evaluation)")

parser.add_argument("--loadrngstate",
        dest="loadrngstate", type=str, required = False,
        help = "Load a random number state file to use for this invocation of the program. NB the treeclassifier uses the random number generator, so only really useful for reproducibilty when running in batch mode")
parser.add_argument("--saverngstate",
        dest = "saverngstate", type=str, required = False,
        help = "Save the random number state at the start of the file.  The inital state is obtained by calling np.random.seed()")
args = parser.parse_args()

if (not args.entergt) and args.extgt is None:
    print "If not entering ground-truth from video frames, external ground truth must be provided"
    sys.exit()

if args.loadrngstate is not None and args.saverngstate is not None:
    print "Can only save OR load rng state"

np.random.seed()
state = np.random.get_state()

if args.saverngstate is not None:
    print "saving initial rng state"
    with open(args.saverngstate, 'wb') as output:
        pickle.dump(state, output, pickle.HIGHEST_PROTOCOL)

if args.loadrngstate is not None:
    print "Loading rng state"
    with open(args.loadrngstate, 'rb') as input:
        state = pickle.load(input)
    np.random.set_state(state)
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
if args.shuffle:
    print "Randomising frames"
    # Must use np.random.shuffle for reproducibility, not shuffle since we have *only* set
    # numpy's random seed
    np.random.shuffle(trainingframes)
else:
    print "Using ordered frames"

if args.extgt is not None:
    print "Loading external ground-truth file"
    externalGT = loadExternalGroundTruth(args.extgt)


trainedframescount = 0


groundtruth = []

if args.entergt:
    cv2.namedWindow("Classification")
    while trainedframescount < len(trainingframes):
        thisframe = trainingframes[trainedframescount]
        img = getVideoFrame(videoFile,thisframe) 
 
        cv2.imshow("Classification", img)
 
        key =  cv2.waitKey(0) 
        if(chr(key) == 'c'):
            localpreds = runClassifier(groundtruth[:(trainedframescount/2)],
                    trackingData.loc[trainingframes[:(trainedframescount/2)]],
                    groundtruth[(trainedframescount/2):],
                    trackingData.loc[trainingframes[(trainedframescount/2):trainedframescount]])


            if args.outfilelocalpreds is not None:
                savePredictions(localpreds,  trackingData, trainingframes[(trainedframescount/2):trainedframescount],args.outfilelocalpreds)

            if args.extgt is not None:
                print "Classification using all remaining external ground truth data:"
                externalpreds = runClassifier(groundtruth[:(trainedframescount/2)],
                        trackingData.loc[trainingframes[:(trainedframescount/2)]],
                        externalGT.loc[trainingframes[trainedframescount:],"state"],
                        trackingData.loc[trainingframes[trainedframescount:]])


                if args.outfileexternalpreds is not None:
                    savePredictions(externalpreds,  trackingData,
                            trainingframes[trainedframescount:], args.outfileexternalpreds)
        elif(chr(key) == 'q'):
            print "Exiting"
            sys.exit()
        elif(chr(key) == 'u'):
            print "Undoing"
            groundtruth.pop()
        else: 
            # TODO check numeric and trap
            groundtruth.append(int(chr(key)))
            if args.extgt is not None:
                print int(externalGT.loc[thisframe])

        trainedframescount = len(groundtruth) 
        print str(trainedframescount) + " frames classified"

else:
    if args.externaltrainingframes is not None:
        trainedframescount = args.externaltrainingframes
    else:
        trainedframescount = int(raw_input("Enter training frames: "))
    groundtruthDF = externalGT.loc[trainingframes[:trainedframescount],"state"]
    groundtruth = list(groundtruthDF)

    localpreds = runClassifier(groundtruth[:(trainedframescount/2)],
                  trackingData.loc[trainingframes[:(trainedframescount/2)]],
                  groundtruth[(trainedframescount/2):],
                  trackingData.loc[trainingframes[(trainedframescount/2):trainedframescount]])

    if args.outfilelocalpreds is not None:
        savePredictions(localpreds,  trackingData, trainingframes[(trainedframescount/2):trainedframescount],args.outfilelocalpreds)
    
    print "Classification using all remaining external ground truth data:"
    externalpreds = runClassifier(groundtruth[:(trainedframescount/2)],
                        trackingData.loc[trainingframes[:(trainedframescount/2)]],
                        externalGT.loc[trainingframes[trainedframescount:],"state"],
                        trackingData.loc[trainingframes[trainedframescount:]])

    if args.outfileexternalpreds is not None:
        savePredictions(externalpreds,  trackingData, trainingframes[trainedframescount:],
                args.outfileexternalpreds)
