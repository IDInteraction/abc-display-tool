#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import unittest

import abcclassify.abcclassify as abcc

""" Classify behaviour in a video file by focussing on areas of poor predictive performance """

class videotracking:

    """ Class containing video data, and truth associated with each frame """
    def __init__(self, videofile=None, framerange=None):
        self.video = None
        self.framerange = None
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

    def trackrange(self):
        return self.framerange


        
class videotrackingTests(unittest.TestCase):
    
    # Unsure how to test exception raised
    # def testLoadNullVideo(self):
    #     self.assertRaises(ValueError, videotracking())
    
    def testLoadVideoRange(self):
        testvid = videotracking(videofile="testvid.mp4")

        self.assertEqual(testvid.trackrange(), range(1,193))

    def testSetRangeWithoutVideo(self):
        testvid = videotracking(framerange=(1,100))

        self.assertEqual(testvid.trackrange(), range(1,100))
    
    def testSetRangeWithVideo(self):
        testvid = videotracking(videofile="testvid.mp4", framerange=(20,30))
        self.assertEqual(testvid.trackrange(), range(20,30))



if __name__ == "__main__":
    unittest.main()
