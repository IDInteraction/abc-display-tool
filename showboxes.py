#!/usr/bin/python

import cv2
import os
import csv
import sys
import numpy as np
import pandas as pd
import itertools
import re
import colorsys


# inputvideo
# output video
# inputcsv arg 3+

bbox_collection = {}

# http://stackoverflow.com/questions/876853/generating-color-ranges-in-python
Nfiles = len(sys.argv) - 3
HSV_tuples = [(x*1.0/Nfiles, 0.5, 0.5) for x in range(Nfiles)]
colours = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
# Must be a bettter way of doing this - scale each colour
ind = 0
for c in colours:
    colours[ind] = tuple(x*255 for  x in c)
    ind = ind + 1


def scaleColour(oldTuple, scalefact):
    # scalefact pre-calculated
    # if scalefact == 1: # pre predictions - make almost black
    #     scalefact = 0.2
    # else:
    #     scalefact = 1/(scalefact - 1) # leave 2 unchanged, half brightness for 2

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

def readFile(indata):
    
    print(os.path.getsize(infile))
    if(os.path.getsize(infile) ==0):
        print(infile + " is size 0; aborting")
        sys.exit()

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
        outdata=pd.read_csv(infile, sep = ",", header = 0, index_col = 0,
                   dtype = {'Frame':np.int32},
                   names = fullformat_names[:num_cols])

    else:
        print("Reading narrow data")
        narrowdata=pd.read_csv(infile, sep = ",", header = None, index_col = 0,
               dtype = {'Frame':np.int32},
               names = narrowformat_names[:num_cols])
        outdata=convertToWide(narrowdata)

    # Add dummy prediction column if it doesn't exist
    if 'pred' not in outdata:
        print("Adding dummy pred column for " + infile)
        outdata["pred"] = 1
    return outdata

###############################################

# Check we're trying to read/write something at least called an mp4 or avi

vidFileRegex = r"(\w+)\.(avi|mp4)$"

if not(re.search(vidFileRegex, sys.argv[1]) and \
        re.search(vidFileRegex, sys.argv[2])):
    print("Invalid input and/or output file format")
    quit()


fileind = 0
for infile in sys.argv[3:]:

    bbox_collection[fileind] = readFile(infile)
    # Recode predicions to give sensible colour intensity ranges
    # Want full brightness if pred is the same for all frameskip
    maxclass = max(bbox_collection[fileind]["pred"])
    minclass = min(bbox_collection[fileind]["pred"])
    if maxclass == minclass:
        bbox_collection[fileind]["colourscale"] = 1.0
    else:
        numclasses= maxclass - minclass
        bbox_collection[fileind]["colourscale"] = bbox_collection[fileind]["pred"]/numclasses
        if(min(bbox_collection[fileind]["colourscale"]) < 0 or \
            max(bbox_collection[fileind]["colourscale"]) > 1):
            print("Colour scaling went wrong; check numbering of preciction classes")
            quit()


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


print("Opening output vid")
fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
videoout = cv2.VideoWriter(sys.argv[2], fourcc, fps,  (ow, oh))
#cv2.namedWindow(WINDOW_NAME)
print("Output vid opened")
frame = 1
got, img = video.read()

while got:
    sys.stdout.write("                      \r" + str(frame))

    lineno = 0
    for infile in sys.argv[3:]:
        cv2.putText(img, infile, (20, 20 + lineno * 18), cv2.cv.CV_FONT_HERSHEY_DUPLEX, 0.5, colours[lineno], 2)
        lineno = lineno + 1


    for bbk in bbox_collection.keys():
        if frame in bbox_collection[bbk].index:
            sys.stdout.write(" " +  str(bbk))
            actbb = bbox_collection[bbk].loc[frame]

            #Need to draw rectangle as four lines, since may not have rotation = 0
            for i in range(1,5):
                j = (i % 4) + 1
                p1 = (actbb['bb' + str(i) + 'x'], actbb['bb' + str(i) + 'y'])
                p2 = (actbb['bb' + str(j) + 'x'], actbb['bb' + str(j) + 'y'])
                p1 = tuple(map(int, p1))
                p2 = tuple(map(int, p2))
                cv2.line(img, p1, p2,  color =scaleColour(colours[bbk], actbb["colourscale"]),  \
                     thickness = 2)



   # cv2.imshow(WINDOW_NAME, img)
    videoout.write(img)


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    got, img = video.read()
    frame = frame + 1
sys.stdout.write("\n")
video.release()
videoout.release()
cv2.destroyAllWindows()
