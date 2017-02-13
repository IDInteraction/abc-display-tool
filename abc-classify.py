#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import argparse
import cv2
import re
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
import pickle
import colorsys

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
        if set(['Timestamp (ms)',
            'Active points',
            'Bounding box centre X (px)']).issubset(indata.columns.values):
            print("CppMT input detected")

            indata.index.names = ['frame']
            del indata['Timestamp (ms)']

        else:
            print("Could not recognise input format")
            sys.exit()

    return indata

def loadExternalGroundTruth(infile, format = "checkfile"):
    if format != "checkfile":
        print "Only checkfiles implemented"
#        sys.exit()

    indata = pd.read_csv(infile, index_col=0,
            names = ["frame", "x", "y", "w", "h", "state"])

    indata.drop(["x","y","w","h"], axis=1 ,inplace = True)

    return indata


def getVideoFrame(videosrc, frameNumber, directFrame = True):
    # Return the video frame from the video
    # Pass in a **1 INDEXED** video frame number
    # This link implies setting frame directly can be problematic, but seems
    # OK on our videos
    # http://stackoverflow.com/questions/11469281/getting-individual-frames-using-cv-cap-prop-pos-frames-in-cvsetcaptureproperty

    if not directFrame:
        fps = videosrc.get(cv2.cv.CV_CAP_PROP_FPS)
        frameTime = 1000 * (frameNumber-1) / fps
        videosrc.set(cv2.cv.CV_CAP_PROP_POS_MSEC, frameTime)
    else:
        videosrc.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameNumber - 1)

    ret, img = videosrc.read()
    if ret == False:
        print "Failed to capture frame " + str(frameNumber - 1)
        sys.exit()

    return img

def getMultiVideoFrame(videosrc, frameNumber):
    # Return the target frame, and the frames either side of it
    # by trial and error 3 frames is about right to see a difference
    # TODO let the user set this value
    # OpenCV returns the 1st frame if we rewind off the beginning;
    # TODO? return a blank frame.  Check what happens at the end of the video
    mainframe = getVideoFrame(videosrc, frameNumber)
    prevframe = getVideoFrame(videosrc, frameNumber - 3 )
    nextframe = getVideoFrame(videosrc, frameNumber + 3)
    h, w = mainframe.shape[:2]

    vis = np.zeros((h, 3*w,3), np.uint8)
    vis[:h, :w, :3] = prevframe
    vis[:h, w:2*w, :3] = mainframe
    vis[:h, 2*w:3*w, :3] = nextframe

    return vis 

def runClassifier(traininggroundtruth, trainingtrackingdata):

    tree = DecisionTreeClassifier() 

    trainedframescount  = len(traininggroundtruth)
    if len(trainingtrackingdata.index) != trainedframescount:
        print "Size mismatch"
        sys.exit()
    print "Classifying with " + str(trainedframescount) + " frames" 

    tree.fit(trainingtrackingdata, traininggroundtruth)

    return tree

def getPredictions(inputtree, alltrackingdata, evaluationframes, groundtruthframes = None, groundtruth = None):

    predictiontrackingdata = alltrackingdata.loc[evaluationframes]
    predictions = inputtree.predict(predictiontrackingdata)

    if groundtruthframes is not None and groundtruth is not None:
        print "Including ground truth in predictions"

        predictions = np.append(predictions,groundtruth)

        evaluationframes = np.append(evaluationframes, groundtruthframes)

    predicted = pd.Series( predictions,  index = evaluationframes)
    
    return predicted


def savePredictions(inputtree,  alltrackingdata, evaluationframes, filename,
        groundtruthframes = None, groundtruth = None):
    # Save the predictions from a tree

    predicted = getPredictions(inputtree, alltrackingdata, evaluationframes)

    predframe = pd.DataFrame(
            {'frame' : evaluationframes, 'attention' : predicted,
                'x' : 200, 'y' : 200, 'w' : 150, 'h' : 150}
            )
    predframe.sort_values("frame",inplace = True)
    predframe.to_csv(filename, index=False, columns=[ "frame",
        "x", "y", "w", "h", "attention"], header=False)

def getAccuracyCrossVal(inputtree, evaluationgroundtruth,
        evaluationtrackingdata):

    try:
        scores = cross_val_score(inputtree,  evaluationtrackingdata, 
                evaluationgroundtruth)
        return  (scores.mean(), scores.std())
    except ValueError:
        print "Cross val accuracy calculation failed"
        return (-1, -1) 

def getShuffledSuccessProbs(inputtree, evaluationgroundtruth, evaluationtrackingdata):
    
    crossvalidation = ShuffleSplit(n_splits = 1000, test_size = 0.5)

    try:
        scores = cross_val_score(inputtree,  evaluationtrackingdata, 
                evaluationgroundtruth, cv=crossvalidation)
        return scores
    except ValueError:
        print "Cross val accuracy calculation failed"
        return (-1, -1) 


def testprob(probs, threshold):
    return sum(probs >= threshold)/float(len(probs))

def getAccuracy(inputtree, groundtruth, trackingdata):

    predicted = inputtree.predict(trackingdata)
    
    accuracy = metrics.accuracy_score(groundtruth, predicted)

    return accuracy

def playbackPredictions(vidsource, predictions, startframe, endframe,
        bbox = (225,125,150,150)):
    
    if endframe - startframe != len(predictions):
            print "Video period of interest is " + chr(endframe - startframe) + " frames, but have predictions for " + chr(len(predictions)) + " frames"


    # http://stackoverflow.com/questions/876853/generating-color-ranges-in-python
    Nstates = max(predictions) + 1

    HSV_tuples = [(x*1.0/Nstates, 0.5, 0.5) for x in range(Nstates)]
    colours = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    # Must be a bettter way of doing this - scale each colour
    ind = 0
    for c in colours:
        colours[ind] = tuple(x*255 for  x in c)
        ind = ind + 1

    cv2.namedWindow("Playback")
    


#   No way to reliably control playback speed - just go for full speed
#   TODO implement something to playback properly
#    fps = vidsource.get(cv2.cv.CV_CAP_PROP_FPS)
#    fudgefactor = 10# Guesstimated amount scale framewait by to get roughly realtim eplayback
#    framewait = int((1000/fps)/fudgefactor)
    framewait = 1
    print framewait
    for f in  range(startframe, endframe):
        vidsource.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f - 1) # frames are 0 indexed
        ret, img = vidsource.read()
        if ret == False:
            print "Failed to capture frame " + str(f - 1)
            break
        try: 
            thispred = predictions.loc[f] 
            cv2.rectangle(img, (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                 color = colours[thispred], # Only have predictions for period of interest
                  thickness = 2 )
        except KeyError:
            print "No predictions for frame: " + f

        textline = 0
        for c in colours:
            cv2.putText(img, "State " + str(textline), (20, 20 + textline * 18), cv2.cv.CV_FONT_HERSHEY_PLAIN, 1.5, c, 2)
            textline = textline + 1


        cv2.imshow("Playback", img)
        key = cv2.waitKey( framewait) & 0xFF
        if chr(key) == 'q':
            break
    print "Press any key to continue"
    cv2.waitKey(0)
    cv2.destroyWindow("Playback")




##############################################

parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video.  For each frame enter a numeric behaviour state.  Press c to classify based on the frames classified so far.  Accuracy is evaluated with cross validation.  Can optionally use an external ground truth file for classification and/or verification.")

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

parser.add_argument("--outfile",
        dest="outfile", type = str, required = False,
        help = "The filename to output classifier performance on the data that haven't been used to construct the classifier") 

parser.add_argument("--loadrngstate",
        dest="loadrngstate", type=str, required = False,
        help = "Load a random number state file to use for this invocation of the program. NB the treeclassifier uses the random number generator, so only really useful for reproducibilty when running in batch mode")
parser.add_argument("--saverngstate",
        dest = "saverngstate", type=str, required = False,
        help = "Save the random number state at the start of the file.  The inital state is obtained by calling np.random.seed()")

parser.add_argument("--summaryfile",
        dest = "summaryfile", type = str, required = False,
        help = "A text file to append summary information from the run to.  Currently records participantCode, trainingframes, startframe, endframe, accuracy (local), accuracy (external)")
parser.add_argument("--participantcode",
        dest = "participantcode", type = str, required = False,
        help = "The participant code to output in the summaryfile")

parser.add_argument("--includegt",
        dest = "includegt", action = "store_true",
        help = "Whether to include ground truth frames when outputting predictions")
parser.set_defaults(includegt=False)


args = parser.parse_args()


if args.summaryfile is not None and args.participantcode is None:
    print "A participant code must be provided if outputting summary data"
    sys.exit()

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
getmulti = False

if args.entergt:
    cv2.namedWindow("Classification")
    while trainedframescount < len(trainingframes):
        thisframe = trainingframes[trainedframescount]
        if getmulti:
            img = getMultiVideoFrame(videoFile, thisframe)
            getmulti = False
        else:
            img = getVideoFrame(videoFile,thisframe) 
 
        cv2.imshow("Classification", img)
 
        key =  cv2.waitKey(0) 
        if(chr(key) == 'c'):
            tree = runClassifier(groundtruth[:(trainedframescount)],
                    trackingData.loc[trainingframes[:(trainedframescount)]])

            (meanAc, stdAc)  = getAccuracyCrossVal(tree, 
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])
            print("Crossval Accuracy: Mean: %0.3f, Std: %0.3f" % (meanAc, stdAc))

            if args.extgt is not None:
                print "Classification using all remaining external ground truth data:"

                predicted = tree.predict(trackingData.loc[trainingframes[trainedframescount:]])
                evaluationgroundtruth = externalGT.loc[trainingframes[trainedframescount:]]
    
                print(metrics.classification_report(evaluationgroundtruth, predicted))
                print(metrics.confusion_matrix(evaluationgroundtruth, predicted))
                print(metrics.accuracy_score(evaluationgroundtruth, predicted))
    

            if args.outfile is not None:
                savePredictions(tree,  trackingData,
                        trainingframes[trainedframescount:], args.outfile)
        elif(chr(key) == 'e'):

            print "Probability accuracy at least:"

            probs = getShuffledSuccessProbs(tree,
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])
            print probs.mean()

            for p in [0.8, 0.9, 0.95, 0.99]:
                print "probability accuracy > " + str(p) + ": " + str(testprob(probs, p))
        elif(chr(key) == 'm'):
            getmulti = True
        elif(chr(key) == 'q'):
            print "Exiting"
            sys.exit()
        elif(chr(key) == 'u'):
            print "Undoing"
            groundtruth.pop()
        elif(chr(key) == 'r'):
            cv2.destroyWindow("Classification")
            print "Playing predictions, including frames manually classified"

            tree = runClassifier(groundtruth[:(trainedframescount)],
                    trackingData.loc[trainingframes[:(trainedframescount)]])

            predictions = getPredictions(tree, trackingData, 
                    trainingframes[trainedframescount:],
                    groundtruthframes = trainingframes[:trainedframescount],
                    groundtruth =  groundtruth)
            playbackPredictions(videoFile, predictions, startVideoFrame, endVideoFrame)
            cv2.namedWindow("Classification")
        else: 
            try:
                groundtruth.append(int(chr(key)))
                if args.extgt is not None:
                    print "External GT was: " + str(int(externalGT.loc[thisframe]))
            except ValueError:
                print "Invalid behaviour state entered; must be numeric"
        trainedframescount = len(groundtruth) 
        print str(trainedframescount) + " frames classified"
        print pd.Series(groundtruth).value_counts()


else:
    if args.externaltrainingframes is not None:
        trainedframescount = args.externaltrainingframes
    else:
        trainedframescount = int(raw_input("Enter training frames: "))
    groundtruthDF = externalGT.loc[trainingframes[:trainedframescount],"state"]
    groundtruth = list(groundtruthDF)

    tree = runClassifier(groundtruth[:(trainedframescount)],
                  trackingData.loc[trainingframes[:(trainedframescount)]])

    (meanAc, stdAc)  = getAccuracyCrossVal(tree, 
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])
    print("Crossval Accuracy: Mean: %0.3f, Std: %0.3f" % (meanAc, stdAc))

    print "Probability accuracy at least:"
    probs = getShuffledSuccessProbs(tree,
        groundtruth[:trainedframescount],
        trackingData.loc[trainingframes[:trainedframescount]])
    print probs.mean()
    

    for p in [0.8, 0.9, 0.95, 0.99]:
        print "probability accuracy > " + str(p) + ": " + str(testprob(probs, p))


    if args.outfile is not None:
        if args.includegt:
            savePredictions(tree,  trackingData, trainingframes[trainedframescount:],
                    args.outfile,
                    groundtruthframes = trainingframes[:trainedframescount],
                    groundtruth = groundtruth[:trainedframescount])
        else:
            savePredictions(tree,  trackingData, trainingframes[trainedframescount:],
                    args.outfile)

 
# TODO - code repetition with accuracy calc
    if args.summaryfile is not None:
        print "Outputting summary"

        with(open(args.summaryfile, 'a')) as summaryfile:
                summaryfile.write(args.participantcode + "," +
                    str(trainedframescount) + "," +
                    str(startVideoFrame) + "," +
                    str(endVideoFrame) + "," + 
                    str(getAccuracyCrossVal(tree,
                        groundtruth[:trainedframescount],
                        trackingData.loc[trainingframes[:trainedframescount]])[1]) + "," +
                    str(getAccuracy(tree,
                        externalGT.loc[trainingframes[trainedframescount:],
                            "state"],
                        trackingData.loc[trainingframes[trainedframescount:]]))
                     + "\n")
                


