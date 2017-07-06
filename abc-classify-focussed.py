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
        return self.framerange

    def addtrackingdata(self, trackingdatafile):
        # Only keep tracking data for the frame range we're interested in
        # TODO Check how many frames we loose - how to allow user to specify threshold?
        thistracking = abcc.loadTrackingData(trackingdatafile)
        self.numtrackingfiles += 1

        filteredtracking = thistracking.loc[self.framerange]

        if self.trackingdata is None:
            self.trackingdata = filteredtracking.copy()
        else:
            self.trackingdata = self.trackingdata.join(thistracking, how="inner", 
             rsuffix="_" + str(self.numtrackingfiles)) 

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

    def testSetRangeWithoutVideo(self):
        testvid = videotracking(framerange=(1,100))

        self.assertEqual(testvid.trackrange(), range(1,100))
    
    def testSetRangeWithVideo(self):
        testvid = videotracking(videofile="./testfiles/testvid.mp4", framerange=(20,30))
        self.assertEqual(testvid.trackrange(), range(20,30))

    def testLoadTrackingData(self):
        testvid = videotracking(framerange=(200,210))
        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        self.assertEqual(testvid.numTrackingFiles(), 1)

        # Check correct size when loading additional tracking data 
        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392 * 2)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        self.assertEqual(testvid.numTrackingFiles(), 2)

        testvid.addtrackingdata("./testfiles/P07_front.openface")
        self.assertEqual(testvid.numTrackingPredictors(), 392 * 3)
        self.assertEqual(testvid.numTrackingFrames(), 10)
        print testvid.trackingdata.columns
        self.assertEqual(testvid.numTrackingFiles(), 3)

if __name__ == "__main__":
    unittest.main()
