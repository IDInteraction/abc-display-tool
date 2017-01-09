#!/usr/bin/python

import cv2
import os
import csv
import sys
import numpy as np
import pandas as pd
import itertools
import re

# inputvideo
# output video
# inputcsv arg 3+

bbox_collection = {}

colours=[(255,0,0),
           (0,255,0),
           (0,0,255),
           (255,255,255)]

def scaleColour(oldTuple, scalefact):
    if scalefact == 1: # pre predictions - make almost black
        scalefact = 0.2
    else:
        scalefact = 1/(scalefact - 1) # leave 2 unchanged, half brightness for 2

    return tuple(x * scalefact for x in oldTuple)

def convertToWide(indata, fps=50):
    # Convert narrow (frame, x,y,w,h) data to wide format
    # i.e. each vertex specified
    indata['bb1x'] = indata['bbx']
    indata['bb1y'] = indata['bby'] + indata['bbh']
    indata['bb2x'] = indata['bbx']
    indata['bb2y'] = indata['bby']
    indata['bb3x'] = indata['bbx'] + indata['bbw']
    indata['bb3y'] = indata['bby']
    indata['bb4x'] = indata['bbx'] + indata['bbw']
    indata['bb4y'] = indata['bby'] + indata['bbh']

    indata['bbr'] = 0

    indata['bbcx'] = indata['bbx'] + indata['bbw'] / 2
    indata['bbcy'] = indata['bby'] + indata['bbh'] / 2

    indata.drop(['bbx','bby'], axis=1, inplace=True)

    return indata


fullformat_names = ["Frame", "time", "actpt", "bbcx", "bbcy",
         "bbw", "bbh", "bbr",
         "bb1x", "bb1y",
         "bb2x", "bb2y",
         "bb3x", "bb3y",
         "bb4x", "bb4y", "pred"]

narrowformat_names = ["Frame", "bbx", "bby", "bbw", "bbh", "pred"]

# Check we're trying to read/write something at least called an mp4 or avi

vidFileRegex = r"(\w+)\.(avi|mp4)$"

if not(re.search(vidFileRegex, sys.argv[1]) and \
        re.search(vidFileRegex, sys.argv[2])):
    print("Invalid input and/or output file format")
    quit()


fileind = 0
for infile in sys.argv[3:]:

    print(os.path.getsize(infile))
    if(os.path.getsize(infile) ==0):
        print(infile + " is size 0; skipping")
        break


    with open(infile) as f:
        print("reading file "  + infile)
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        first_row = next(reader)
        num_cols = len(first_row)

    print str(num_cols) + " columns in file"
    # TODO deal with csv files with row numbers; will throw off prediction test
    if(num_cols == 17 or num_cols == 6):
        print("have predictions")

    elif(num_cols ==16 or num_cols == 5):
        print("no predictions")
    else:
        print("Unknown input format")
        quit()

    if(num_cols == 16 or num_cols ==17):
        print("Reading wide data")
        bbox_collection[fileind]=pd.read_csv(infile, sep = ",", header = 0, index_col = 0,
                   dtype = {'Frame':np.int32},
                   names = fullformat_names[:num_cols])

    else:
        print("Reading narrow data")
        narrowdata=pd.read_csv(infile, sep = ",", header = None, index_col = 0,
               dtype = {'Frame':np.int32},
               names = narrowformat_names[:num_cols])
        bbox_collection[fileind]=convertToWide(narrowdata)

    # Add dummy prediction column if it doesn't exist
    if 'pred' not in bbox_collection[fileind]:
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
            for i in range(1,5):
                j = (i % 4) + 1
                p1 = (actbb['bb' + str(i) + 'x'], actbb['bb' + str(i) + 'y'])
                p2 = (actbb['bb' + str(j) + 'x'], actbb['bb' + str(j) + 'y'])
                p1 = tuple(map(int, p1))
                p2 = tuple(map(int, p2))
                cv2.line(img, p1, p2,  color = scaleColour(colours[bbk], actbb["pred"].astype(float)),  \
                     thickness = 2)

                lineno = 0
            for infile in sys.argv[3:]:
                cv2.putText(img, infile, (20, 20 + lineno * 18), cv2.cv.CV_FONT_HERSHEY_DUPLEX, 0.5, colours[lineno], 1)
                lineno = lineno + 1


   # cv2.imshow(WINDOW_NAME, img)
    videoout.write(img)


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    got, img = video.read()
    frame = frame + 1

video.release()
videoout.release()
cv2.destroyAllWindows()
