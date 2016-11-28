#!/usr/bin/python

import cv2
import os
import csv
import sys
import numpy as np
import pandas as pd
import itertools


bbox_collection = {}

colours =np.array([(255,0,0),
           (0,255,0),
           (0,0,255),
           (255,255,255)])

col_names = ["Frame", "time", "actpt", "bbcx", "bbcy",
         "bbw", "bbh", "bbr",
         "bb1x", "bb1y",
         "bb2x", "bb2y",
         "bb3x", "bb3y",
         "bb4x", "bb4y", "pred"]


fileind = 0
for infile in sys.argv[3:]:

    # Determine whether we have a prediction column

    with open(infile) as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        first_row = next(reader)
        num_cols = len(first_row)

    if(num_cols == 17):
        print("have predictions")
    elif(num_cols ==16):
        print("no predictions")
    else:
        print("Unknown input format")
        quit()

    bbox_collection[fileind]=pd.read_csv(infile, sep = ",", header = 0, index_col = 0,
               dtype = {'Frame':np.int32},
               names = col_names[:num_cols])


    if(num_cols == 16):
        print("Adding dummy pred column for " + infile)
        bbox_collection[fileind]["pred"] = 1

    fileind = fileind + 1

#for bbk in bbox_collection.keys():
    #print bbox_collection[bbk].index
#    print bbox_collection[bbk]

#quit()


#WINDOW_NAME = 'Playback'

video = cv2.VideoCapture(sys.argv[1])

fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
ow = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
oh = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
videoout = cv2.VideoWriter(sys.argv[2], fourcc, fps,  (ow, oh))
#cv2.namedWindow(WINDOW_NAME)

frame = 1
got, img = video.read()

while got:

    for bbk in bbox_collection.keys():
        if frame in bbox_collection[bbk].index:
            print frame, bbk
            actbb = bbox_collection[bbk].loc[frame]

            #Need to draw rectangle as four lines, since may not have rotation = 0
            #cv2.rectangle(img, (actbb['bb1x'].astype(int), actbb['bb1y'].astype(int)),
            #          (actbb['bb3x'].astype(int), actbb['bb3y'].astype(int)),colours[bbk], 2)
            # From Rob's code (note column 0 is used as the index)
            for i in xrange(4):
                n = (i + 4) * 2 - 1
                m = (((i + 1) % 4) + 4) * 2 -1
                p1 = (actbb[n].astype(int), actbb[n+1].astype(int))
                p2 = (actbb[m].astype(int), actbb[m+1].astype(int))
                cv2.line(img, p1, p2, \
                color = colours[bbk], \
                thickness = actbb["pred"].astype(int))


   # cv2.imshow(WINDOW_NAME, img)
    videoout.write(img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    got, img = video.read()
    frame = frame + 1

video.release()
videoout.release()
cv2.destroyAllWindows()
