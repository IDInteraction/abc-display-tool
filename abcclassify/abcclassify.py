#!/usr/bin/env python

""" Classify behaviour in a video file by focussing on areas of poor predictive performance """

import cv2
import numpy as np
import pandas as pd
import unittest
import copy

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

class videotracking(object):

    """ Class containing video data, tracking and classifications associated with each frame """
    def __init__(self, videofile=None, framerange=None, trackingdatafile = None):
        self.video = None
        self.frames = None
        self.trackingdata = None # Contains sources of tracking information (e.g. CppMT, openface data etc.)
        self.classificationdata = None # Contains the behavioural classifications that have been set by the user
        self.classficationmethod = None # the method used to classify the frames (used to prevent us doing xvalidation on non-random classifications)
        self.numtrackingfiles = 0

        if videofile is None and framerange is None:
            print("Must supply a framerange if videofile is none")
            raise ValueError

        if videofile is not None:
            self.video = cv2.VideoCapture(videofile)

        if framerange is not None:
            if len(framerange) != 2:
                print("Video framerange must by a tuple of length 2")
                raise ValueError
            # TODO - test framerange is within video or raise exception
            # TODO - test first frame >=1
            self.frames = range(framerange[0], framerange[1]) 
        else:
            lastframe = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self.frames = range(1, lastframe + 1)

        # We can't store NaNs in integer numpy arrays, so we use -1 for missing    
        self.classificationdata = pd.Series(data = [-1] * len(self.frames),
            index = self.frames, dtype = np.int64)

        if trackingdatafile is not None:
            self.trackingdata = addtrackingdata(trackingdatafile)

    def split(self, framerange):
        """ Return a videotracking object containing a contiguous subset of frames """
        # TODO subset rather than copy and delete
        newvt = copy.deepcopy(self)
        # newvt.video - this will (potentially) contain more frames than we want
        if len(framerange) != 2:
            raise ValueError("framerange must be a tuple of length 2")
        if framerange[0] < min(newvt.frames) or framerange[1] > max(newvt.frames):
            raise ValueError("can only split on frames that already exist")

        # TODO handle splitting if we've missing frames in the range
        newvt.frames = range(framerange[0], framerange[1])
        newvt.trackingdata = newvt.trackingdata.loc[newvt.frames]
        newvt.classificationdata = newvt.classificationdata.loc[newvt.frames]

        return newvt

    def join(self, extravt):
        """ Join the data in extravt to self, and return this in a new object
        leaving self untouched """

        # check video is the same in both objects
        if self.video != extravt.video:
            raise ValueError("Source video file is different")
        
        newvt = copy.deepcopy(self)
        # check frame ranges don't overlap 

        if len(set(self.frames) & set(extravt.frames)) > 0:
            raise ValueError("Frames must not overlap")

        # Check same tracking data (same columns) in both parts
        if newvt.numtrackingfiles != extravt.numtrackingfiles:
            raise ValueError("Both sources must have the same number of tracking files")

        # do the join
        newvt.classificationdata = newvt.classificationdata.append(extravt.classificationdata)
        newvt.trackingdata = newvt.trackingdata.append(extravt.trackingdata)
        newvt.frames += extravt.frames

        # Sort everything in frame order
        newvt.classificationdata.sort_index(inplace=True)
        newvt.trackingdata.sort_index(inplace=True)
        newvt.frames.sort()

        # Check we have tracking data and (possibly empty) classifications for each frame
        if list(newvt.classificationdata.index) != list(newvt.trackingdata.index) or \
            list(newvt.trackingdata.index) != newvt.frames:
            raise ValueError("Error when joining objects")

        return newvt



    def trackrange(self):
        """ Return the extent of frames we (aim) to generate predictions for"""
        return self.frames 

    def gettrackableframes(self):
        """" Return frames that we have tracking data for """
        if self.numtrackingfiles == 0:
            return None
        elif len(self.trackingdata.index) == 0:
            return None
        else:
             return list(self.trackingdata.index)
    

    def loadTrackingData(self, infile, guessClean=True):
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

    def addtrackingdata(self, trackingdatafile):
        # TODO Check how many frames we loose - how to allow user to specify threshold?
        thistracking = self.loadTrackingData(trackingdatafile)
        self.numtrackingfiles += 1

        # A KeyError is thrown if we don't have tracking for all the frames in framerange
        # So we first get the subset of frames that are in the tracking data before filtering down
        trackingframes = set(thistracking.index).intersection(set(self.frames))

        if len(trackingframes) > 0:
            filteredtracking = thistracking.loc[trackingframes]
        else:
            print "Warning: no matching frames found in tracking data"
            filteredtracking = thistracking[0:0]

        if self.trackingdata is None:
            self.trackingdata = filteredtracking.copy()
        else:
            self.trackingdata = self.trackingdata.join(thistracking, how="inner", 
             rsuffix="_" + str(self.numtrackingfiles)) 
        self.trackableframes = list(self.trackingdata.index.values)

    def numTrackingPredictors(self):
        return len(self.getTrackingColumns())

    def numTrackingFrames(self):
        return len(self.trackingdata.index)
    
    def numTrackingFiles(self):
        return self.numtrackingfiles

    def getTrackingColumns(self):
        return list(self.trackingdata.columns)
    
    def setClassification(self, frame, state):
        if self.classficationmethod is None:
            raise ValueError("Must specify classification method before classifying frames")
        """ Set the behaviour classification for a frame"""
        if state == -1:
            raise ValueError("Cannot set behaviour to be -1")
        if state is None:
            state = -1

        updateseries = pd.Series(data = [state], index=[frame])

        # Test the frame we're trying to update exists
        if not set(updateseries.index).issubset(set(self.classificationdata.index)):
            raise ValueError("Attempted to update classification for a frame that does not exist")

        self.classificationdata.update(updateseries)

    def setClassificationMethod(self, method):
        classificationmethods = ["random", "sequential"] 
        if not (method in classificationmethods):
            raise ValueError("Classification method must be one of " + str(classificationmethods))
        self.classficationmethod = method

    def getClassificationMethod(self):
        return self.classficationmethod

    def getClassification(self, frame):
        thisclassification =  self.classificationdata[frame]
        if thisclassification == -1:
            return None
        else:
            return thisclassification

    def getClassifiedFrames(self):
        classifiedframes = self.classificationdata.loc[self.classificationdata != -1]
        return classifiedframes

    def getTrackingForClassifiedFrames(self):
        tframes = self.trackingdata.loc[self.getClassifiedFrames().index]
        return tframes

    def getTrackingForUnclassifiedFrames(self):
        untrackinds = set(self.gettrackableframes()) - set( self.getClassifiedFrames().index)
        return self.classificationdata.loc[untrackinds] 

    def getTrackingForFrames(self, indices):
        return self.trackingdata.loc[indices]

    def numClassifiedFrames(self):
        numclassified = len(self.getClassifiedFrames())
        return numclassified
    
class videotrackingclassifier(object):
    """ Fit a (decision tree) classifier to a videotracking object
    
    This is a wrapper to the sklearn code, which pulls out the appropriate frames to
    run the classier on """

    def __init__(self, videoTrackingObject, random_state = None):
        self.classifier = DecisionTreeClassifier(random_state = random_state)
        self.vto = videoTrackingObject
        self.classifier.fit(self.vto.getTrackingForClassifiedFrames(), self.vto.getClassifiedFrames())

    def getPredictions(self, frames):
        if not set(frames).issubset(set(self.vto.gettrackableframes())):
            raise ValueError("Trying to predict for frames without tracking data")

        if len(set(frames) & set(self.vto.getClassifiedFrames().index)) > 0:
            raise ValueError("Trying to predict for frames that have already been classified")

        trackingdata = self.vto.getTrackingForFrames(frames)

        preds = self.classifier.predict(trackingdata)
        return preds

    def getMetric(self, truth, metric):
        preds = self.getPredictions(truth.index)

        metric = metric(truth, preds)
        return metric

    def getAccuracy(self, truth):
        """ Shortcut to get the accuracy """
        accuracy = self.getMetric(truth, metrics.accuracy_score )
        return accuracy

    def getCrossValidatedScore(self):
        if self.vto.getClassificationMethod() != "random":
            raise ValueError("Cross validation is only meaningful when frames have been classified at random")
        
        score = cross_val_score(self.classifier, \
            self.vto.getTrackingForClassifiedFrames(),
            self.vto.getClassifiedFrames())

        return score

