#!/usr/bin/python

import cv2
import os
import csv
import sys
import numpy as np
import pandas as pd
# Want to read in all arguments >2 and put the csv file from each into a dictionary

df=pd.read_csv(sys.argv[3], sep = ",", header = 0, index_col = 0,
               dtype = {'Frame':np.int32},
               names = ["Frame", "time", "actpt", "bbcx", "bbcy",
                        "bbw", "bbh", "bbr",
                        "bb1x", "bb1y",
                        "bb2x", "bb2y",
                        "bb3x", "bb3y",
                        "bb4x", "bb4y"])


WINDOW_NAME = 'Playback'

video = cv2.VideoCapture(sys.argv[1])

fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
ow = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
oh = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
videoout = cv2.VideoWriter(sys.argv[2], fourcc, fps,  (ow, oh))
cv2.namedWindow(WINDOW_NAME)

frame = 1
got, img = video.read()

while got:
    
    
    if frame in df.index:
        print frame
        actbb = df.loc[frame]
        cv2.rectangle(img, (actbb['bb2x'].astype(int), actbb['bb2y'].astype(int)),
                      (actbb['bb2x'].astype(int) + actbb['bbw'].astype(int),
                       actbb['bb2y'].astype(int) + actbb['bbh'].astype(int)), (255, 0, 0), 2)  

    cv2.imshow(WINDOW_NAME, img)


    videoout.write(img)

        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    got, img = video.read()
    frame = frame + 1
    
video.release()
videoout.release()
cv2.destroyAllWindows()

