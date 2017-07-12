#!/usr/bin/python
import csv
import os
import sys
import numpy as np
import pandas as pd
import re
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pickle
import colorsys

def loadTrackingData(infile, guessClean=True):
    print "Loading tracking data from: " + infile
    indata=pd.read_csv(infile, index_col=0)
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
        elif set(['Timestamp (ms)',
            'Active points',
            'Bounding box centre X (px)']).issubset(indata.columns.values):
            print("CppMT input detected")

            indata.index.names = ['frame']
            del indata['Timestamp (ms)']
        else:
            print("Could not recognise input format")
            print("Assuming frame is column 0")
            
            indata = pd.read_csv(infile, index_col = 0)

            if indata.index.name is None:
                print "Using unnamed column as frame"
            else:
                print "Using " + indata.index.name + " as frame"
            print "Using the following columns for classifier:"
            print indata.columns.values
            

    return indata

def loadExternalGroundTruth(infile, format="checkfile"):
    if format != "checkfile":
        print "Only checkfiles implemented"
#        sys.exit()
    with open(infile, 'rb') as csvfile:
        hasheader=csv.Sniffer().has_header(csvfile.read(2048))
        

    if hasheader:
        indata = pd.read_csv(infile, index_col=0, header = 0,
            names = ["frame", "x", "y", "w", "h", "state"])
    else:
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

def runClassifier(traininggroundtruth, trainingtrackingdata, trees):

    if trees == 1:
        decisionTree = DecisionTreeClassifier()
    else:
        decisionTree = RandomForestClassifier(n_estimators = trees)

    trainedframescount  = len(traininggroundtruth)
    if len(trainingtrackingdata.index) != trainedframescount:
        print "Size mismatch"
        sys.exit()
    print "Classifying with " + str(trainedframescount) + " frames"
    decisionTree.fit(trainingtrackingdata, traininggroundtruth)

    return decisionTree

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
        # Since np.random.set_state() isn't thread safe we cannot use
        # n_jobs > 1 on cross_val_score, since this will break reproducibility
        scores = cross_val_score(inputtree,  evaluationtrackingdata,
                evaluationgroundtruth)
        return  (scores.mean(), scores.std(), np.percentile(scores, 2.5),
                np.percentile(scores, 97.5))
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

def getF1Score(inputtree, groundtruth, trackingdata):
    predicted = inputtree.predict(trackingdata)
    f1score = metrics.f1_score(groundtruth, predicted)

    return f1score

# Callback for when trackbar changes position
def onChange(trackbarValue):
    pass

def playbackPredictions(vidsource, predictions, startframe, endframe,
        bbox = (225,125,150,150)):


    if endframe - startframe != len(predictions):
            print "Video period of interest is " + str(endframe - startframe) + " frames, but have predictions for " + str(len(predictions)) + " frames"


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

    cv2.createTrackbar("position", "Playback", startframe, endframe - 1, onChange)


    framewait = 1

    ret = True
    f = startframe
    while ret:
        ret, img = vidsource.read()
        if ret == False:
            print "Failed to capture frame " + str(f - 1)
            break

        f = cv2.getTrackbarPos("position", "Playback")
        vidsource.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f)
        try:
            thispred = predictions.loc[f]
            cv2.rectangle(img, (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                 color = colours[thispred],
                  thickness = 2 )
        except KeyError: # Didn't have prediction for frame
            pass

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

def getTableSuffix(filename):
    """ Extract just the filename from the filepath, and only keep alphanumeric 
    characters """
    filename = re.sub('[\W]+', '', os.path.basename(filename))

    return filename

