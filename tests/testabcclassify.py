#!/usr/bin/env python
import unittest
import pandas as pd

from .context import abcc

class videotrackingTests(unittest.TestCase):
    
    # Unsure how to test exception raised
    # def testLoadNullVideo(self):
    #     self.assertRaises(ValueError, videotracking())
    
    def testLoadVideoRange(self):
        testvid = abcc.videotracking(videofile="./testfiles/testvid.mp4")

        self.assertEqual(testvid.trackrange(), range(1,193))
        self.assertIsNone(testvid.gettrackableframes())

    def testSetRangeWithoutVideo(self):
        testvid = abcc.videotracking(framerange=(1,100))
        self.assertEqual(testvid.trackrange(), range(1,100))
        self.assertIsNone(testvid.gettrackableframes())
    
    def testSetRangeWithVideo(self):
        testvid = abcc.videotracking(videofile="./testfiles/testvid.mp4", framerange=(20,30))
        self.assertEqual(testvid.trackrange(), range(20,30))
        self.assertIsNone(testvid.gettrackableframes())
    
    def testRangeWithinVideo(self):
        self.assertRaises(ValueError, abcc.videotracking, videofile="./testfiles/testvid.mp4", framerange=(0,10))
        self.assertRaises(ValueError, abcc.videotracking, videofile="./testfiles/testvid.mp4", framerange=(1,1000))

    def testLoadTrackingData(self):
        testvid = abcc.videotracking(framerange=(200,210))

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
        testvid2 = abcc.videotracking(framerange=(1,25))
        testvid2.addtrackingdata("./testfiles/P07firstframes.openface")
        self.assertEqual(testvid2.numTrackingFrames(), 19)
        self.assertEqual(testvid2.gettrackableframes(), range(1,20))

    def testClassifyingFrames(self):
        testvid = abcc.videotracking(framerange=(1,25))
        testvid.addtrackingdata("./testfiles/P07firstframes.openface")
        
        self.assertRaises(ValueError, testvid.setClassificationMethod, "invalid")

        testvid.setClassificationMethod("sequential")

        self.assertEqual(testvid.getClassificationMethod(), "sequential")

        # Check we can provide a classification for a frame, and retrieve it
        testvid.setClassification(2,1)
        testvid.setClassification(5,0)

        # Check we can return classified frames, and their classification data
        self.assertEqual(testvid.numClassifiedFrames(),2)
        frames = testvid.getClassifiedFrames()
        self.assertEqual(list(frames.index), [2,5])
        self.assertEqual(list(frames), [1,0])

        self.assertRaises(ValueError, testvid.setClassification,2,0, testunset = True)

        # Check we can return unclassified frames
        uframes = testvid.getTrackingForUnclassifiedFrames()
        self.assertEqual(set(testvid.gettrackableframes()) - set(frames.index), set(uframes.index))
        self.assertEqual(set(frames.index) | set(uframes.index), set(testvid.gettrackableframes()))

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

        # Fit a decision tree classifier to the classified frames
            
        testvid = abcc.videotracking(videofile="./testfiles/testvid.mp4")
        testvid.addtrackingdata("./testfiles/P07firstframes.openface")
        testvid.setClassificationMethod("sequential")

        for i in range(1,6):
            testvid.setClassification(i,1)
        for i in range(6,11):
            testvid.setClassification(i,0)

        # This builds a dt classifier for all frames we've classified
        dtclass = abcc.videotrackingclassifier(testvid, random_state=123)

        # Get accuracy statistics for fitted model based on external ground truth
        gtdata = ([1] * 5) + ([0] * 4)
        groundtruth = pd.Series(data = gtdata, index = range(11,20))

        self.assertEqual(dtclass.getAccuracy(groundtruth), 4.0/9.0)

        excessgroundtruth = pd.Series(data = gtdata + gtdata, index = range(11,29))
        self.assertRaises(ValueError, dtclass.getAccuracy, excessgroundtruth)

        # Should fail if we try and calculate accuracy using frames we've already classified
        overlapgt = pd.Series(data = gtdata, index = range(10,19))
        self.assertRaises(ValueError, dtclass.getAccuracy, overlapgt)

        # Can only get cross-val accuracy from the object, since it doesn't know about ground truth
        # but cross val accuracy only makes sense if we've classified frames at random
        # (cannot test this within the object - it only knows which frames have been classified -
        # not how they were selected)

        self.assertRaises(ValueError, dtclass.getCrossValidatedScore)

        # Pretend we've classified at random to test cross val
        testvid.setClassificationMethod("random")
        dtrand = abcc.videotrackingclassifier(testvid, random_state=123)
        meanscore = dtrand.getCrossValidatedScore().mean()
        self.assertEqual(meanscore, 0.25)

    def testSplittingAndJoiningObject(self):
        testvid = abcc.videotracking(framerange=(1,25))
        testvid.addtrackingdata("./testfiles/P07_front.openface")
        testvid.setClassificationMethod("sequential")

        for i in range(1,6):
            testvid.setClassification(i,1)
        for i in range(10,18):
            testvid.setClassification(i,0)

        # should be able to extract a contiguious subset of frames from the object and 
        # return in a new object
        splitframes = (5,10) 
        splitframerange = range(splitframes[0], splitframes[1])
        splitvid = testvid.split(splitframes)

        self.assertEqual(splitvid.gettrackableframes(), splitframerange)
        self.assertEqual(list(splitvid.getTrackingForFrames(splitframerange).index), splitframerange )

        # Should fail if we try and subset on frames we don't have 
        self.assertRaises(ValueError, testvid.split, (3,30))

        splitframes2 = (10, 15)
        splitvid2 = testvid.split(splitframes2)

        joinvid = splitvid.join(splitvid2)
        # TODO check we've not touched splitvid

        self.assertEqual(joinvid.gettrackableframes(), range(5,15))
        
        # Check we can join in either order
        joinvid2 = splitvid2.join(splitvid)
        self.assertEqual(joinvid.gettrackableframes(),joinvid2.gettrackableframes())

        splitframes3 = (8, 15)
        splitvid3 = testvid.split(splitframes3)

        # Exception should be raised if we try and join with an overlap
        self.assertRaises(ValueError, testvid.join, splitvid3)

        # How do we handle gaps when joining???

        # Check we've copied the object and not just a reference to it
        testframe = 7
        self.assertIsNone(testvid.getClassification(testframe))
        self.assertIsNone(splitvid.getClassification(testframe))
        testvid.setClassification(testframe, 2)
        self.assertEqual(testvid.getClassification(testframe), 2)
        self.assertIsNone(splitvid.getClassification(testframe))

if __name__ == "__main__":
    unittest.main()
    
