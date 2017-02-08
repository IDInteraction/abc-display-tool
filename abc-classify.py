#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import argparse
import cv2
import re

# Need video file
# Openface / cppmt data

# Random sample of frames
# Display an image and get prediction


# Construct rpart tree
# Use rpart tree to predict

def loadTrackingData(infile,
        guessClean = True):
    indata = pd.read_csv(infile)
    stripvals = [x.strip(' ') for x in indata.columns.values]
    indata.columns = stripvals

    if guessClean == True:
        # Remove spurious columns based on filetype
        if set(['frame', 'timestamp', 'gaze_0_x']).issubset(indata.columns.values):
            print("OpenFace input detected")
            AUCols = [x.find("AU") == 0 for x in indata.columns.values] 
            ControlCols = [True] * 4 + [False] * (indata.columns.values.size - 4)
            mask = [x|y for (x,y) in zip(AUCols, ControlCols)] 

            indata.drop(indata.columns[mask], axis = 1, inplace = True)

    return indata




parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video")

parser.add_argument("--videofile",
        dest = "videofile", type = str, required = True)
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True)
parser.add_argument("--startframe",
        dest = "startframe", type = int, required = False)
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False)

args = parser.parse_args()

video_file = cv2.VideoCapture(args.videofile)

trackingData = loadTrackingData(args.trackerfile)


print(trackingData)


