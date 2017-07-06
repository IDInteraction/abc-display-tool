#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import unittest

import abcclassify.abcclassify as abcc

""" Classify behaviour in a video file by focussing on areas of poor predictive performance """

class videotracking:

    """ Class containing video data, tracking and classifications associated with each frame """
    def __init__(self, videofile=None, framerange=None, trackingdatafile = None):
        self.video = None
        self.framerange = None
        self.trackingdata = None
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
        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        self.assertEqual(testvid.numTrackingFiles(), 1)
        self.assertEqual(testvid.gettrackableframes(), range(200,210))

        # Check correct size when loading additional tracking data 
        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392 * 2)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        self.assertEqual(testvid.numTrackingFiles(), 2)
        self.assertEqual(testvid.gettrackableframes(), range(200,210))

        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392 * 3)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        self.assertEqual(testvid.numTrackingFiles(), 3)
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

        # Test we have tracking data for all the frames left in the videotracking object
#        self.assertEqual(testvid2.)



if __name__ == "__main__":
    unittest.main()
