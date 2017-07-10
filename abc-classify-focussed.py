#!/usr/bin/env python

""" Classify behaviour in a video file by focussing on areas of poor predictive performance """

import cv2
import numpy as np
import pandas as pd
import unittest
import abcclassify.abcclassify as abcc

from sklearn.tree import DecisionTreeClassifier


class videotracking:

    """ Class containing video data, tracking and classifications associated with each frame """
    def __init__(self, videofile=None, framerange=None, trackingdatafile = None):
        self.video = None
        self.framerange = None
        self.trackingdata = None # Contains sources of tracking information (e.g. CppMT, openface data etc.)
        self.classificationdata = None # Contains the behavioural classifications that have been set by the user
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
            self.framerange = range(framerange[0], framerange[1]) 
        else:
            lastframe = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self.framerange = range(1, lastframe + 1)
        # We can't store NaNs in integer numpy arrays, so we use -1 for missing    
        self.classificationdata = pd.Series(data = [-1] * len(self.framerange),
            index = self.framerange, dtype = np.int64)

        if trackingdatafile is not None:
            self.trackingdata = addtrackingdata(trackingdatafile)

    def trackrange(self):
        """ Return the extent of frames we (aim) to generate predictions for"""
        return self.framerange

    def gettrackableframes(self):
        """" Return frames that we have tracking data for """
        if self.numtrackingfiles == 0:
            return None
        elif len(self.trackingdata.index) == 0:
            return None
        else:
             return list(self.trackingdata.index)
    
    def addtrackingdata(self, trackingdatafile):
        # TODO Check how many frames we loose - how to allow user to specify threshold?
        thistracking = abcc.loadTrackingData(trackingdatafile)
        self.numtrackingfiles += 1

        # A KeyError is thrown if we don't have tracking for all the frames in framerange
        # So we first get the subset of frames that are in the tracking data before filtering down
        trackingframes = set(thistracking.index).intersection(set(self.framerange))

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
        return len(self.trackingdata.columns)

    def numTrackingFrames(self):
        return len(self.trackingdata.index)
    
    def numTrackingFiles(self):
        return self.numtrackingfiles
    
    def setClassification(self, frame, state):
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

    def numClassifiedFrames(self):
        numclassified = len(self.getClassifiedFrames())
        return numclassified
    
# class videotrackingClassifier:
#     """ Fit a (decision tree) classifier to a videotracking object
    
#     This is a wrapper to the sklearn code, which pulls out the appropriate frames to
#     run the classier on """

#     def __init__(self, videotrackingObject):
#         self.classifier = DecisionTreeClassifier()


        
#         self.classifier.fit(tracking, classified frames)
        


class videotrackingTests(unittest.TestCase):
    
    # Unsure how to test exception raised
    # def testLoadNullVideo(self):
    #     self.assertRaises(ValueError, videotracking())
    
    def testLoadVideoRange(self):
        testvid = videotracking(videofile="./testfiles/testvid.mp4")

        self.assertEqual(testvid.trackrange(), range(1,193))
        self.assertIsNone(testvid.gettrackableframes())

    def testSetRangeWithoutVideo(self):
        testvid = videotracking(framerange=(1,100))
        self.assertEqual(testvid.trackrange(), range(1,100))
        self.assertIsNone(testvid.gettrackableframes())
    
    def testSetRangeWithVideo(self):
        testvid = videotracking(videofile="./testfiles/testvid.mp4", framerange=(20,30))
        self.assertEqual(testvid.trackrange(), range(20,30))
        self.assertIsNone(testvid.gettrackableframes())

    def testLoadTrackingData(self):
        testvid = videotracking(framerange=(200,210))

        # Test we can load files
        # and 
        # Check correct size when loading additional tracking data 
        for i in range(1,4):
            testvid.addtrackingdata("./testfiles/P07_front.openface")
            self.assertEqual(testvid.numTrackingPredictors(), 392 * i)
            self.assertEqual(testvid.numTrackingFrames(), 10)
            self.assertEqual(testvid.numTrackingFiles(), i)
            self.assertEqual(testvid.gettrackableframes(), range(200,210))

        
        # Check we loose frames when loading missing tracking data
        # No overlap on this one
        testvid.addtrackingdata("./testfiles/P07firstframes.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392 * 4)
        self.assertEqual(testvid.numTrackingFrames(), 0)
        self.assertEqual(testvid.numTrackingFiles(), 4)
        self.assertIsNone(testvid.gettrackableframes())

        # partial overlap
        testvid2 = videotracking(framerange=(1,25))
        testvid2.addtrackingdata("./testfiles/P07firstframes.openface")
        self.assertEqual(testvid2.numTrackingFrames(), 19)
        self.assertEqual(testvid2.gettrackableframes(), range(1,20))

    def testClassifyingFrames(self):
        testvid = videotracking(framerange=(1,25))
        testvid.addtrackingdata("./testfiles/P07firstframes.openface")

        # Check we can provide a classification for a frame, and retrieve it
        testvid.setClassification(2,1)
        testvid.setClassification(5,0)

        # Check we can return classified frames, and their tracking data
        self.assertEqual(testvid.numClassifiedFrames(),2)
        frames = testvid.getClassifiedFrames()
        self.assertEqual(list(frames.index), [2,5])
        self.assertEqual(list(frames), [1,0])

        trackframes = testvid.getTrackingForClassifiedFrames()
        self.assertEqual(list(trackframes.index), [2,5])
        self.assertEqual(list(trackframes["pose_Tx"]), [64.9382, 63.9397])

        self.assertEqual(testvid.getClassification(2),1)
        self.assertEqual(testvid.getClassification(5),0)        

        #Unclassified frames should return None
        self.assertIsNone(testvid.getClassification(1))

        # We should only be able to classify frames we have tracking data for
        self.assertRaises(ValueError, testvid.setClassification, 100, 0)

        # We should be able to change the classification of a frame
        testvid.setClassification(2,1)
        self.assertEqual(testvid.getClassification(2),1)

        # We should be able to unclassify a frame  - for completeness
        testvid.setClassification(2,None)
        self.assertIsNone(testvid.getClassification(2))

        # Should not be able to set a classification that's our internal missing
        self.assertRaises(ValueError, testvid.setClassification, 100, -1)

    def testFittingModel(self):
        pass
        
        # Fit a decision tree classifier to the classified frames
        # Would we ever want to do this on a subset of the classified frames?


        # Get accuracy statistics for fitted model

        # Get other measures of accuracy (e.g. f1) statistics for fitted model

        # Can only get cross-val accuracy from the object, since it doesn't know about ground truth
        # but cross val accuracy only makes sense if we've classified frames at random
        # (cannot test this within the object - it only knows which frames have been classified -
        # not how they were selected)

        # Should ground truth go in the object? (if it exists)
        # pros: will allow accuracy calculations for classifier within object
        # cons: more complexity as we don't always have ground truth
        # need a loadGroundTruth() method

        # Can have >1 measure of accuracy from a videotracking object (i.e. xval or ground truth, f1 or accuracy 
        # (or arbitrary metric of model performance))
        

    def testSplittingObject(self):
        pass

        # should be able to extract an arbitrary subset of frames from the object and 
        # return in a new object
        
        # Should be able to join arbitrary subsets of frames and return composite object

        # (Should include tracking and classification data for the frames)




if __name__ == "__main__":
    unittest.main()
