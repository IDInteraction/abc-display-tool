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
from sklearn.preprocessing import MinMaxScaler

# inputvideo
# output video
# inputcsv arg 3+

bbox_collection = {}
barchartOffsets = [100, 300]

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

analogue_names = ["Frame", "c1", "c2", "c3"]

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
    # Test if we have digits in the first row
    if len([s for s in first_row if s.isdigit()]) != 0:
        print "No header row detected"
        print "Ensure file contains appropriately named columns:"
        hasHeaderRow = False 
    else:
        print "Detected header row"
        hasHeaderRow = True

    if not hasHeaderRow: # Assume legacy bbox file
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

            print "converting to wide"
            outdata=convertToWide(narrowdata)
            
        trackingType = "bbox"

    else: # We have a header row
        outdata = pd.read_csv(infile, sep = ",", index_col = 0, dtype = {'Frame':np.int32})
        # Check whether we have bbox format data
        colnames = set(list(outdata))
        colnames.add("Frame")

        if colnames.issuperset(["bbx", "bby", "bbw", "bbh"]):
            print "Converting to wide"
            outdata = convertToWide(outdata)

    if getTrackType(outdata) == "bbox":
        # Add dummy prediction column if it doesn't exist
        if 'pred' not in outdata:
            print("Adding dummy pred column for " + infile)
            outdata["pred"] = 1
        # Recode predicions to give sensible colour intensity ranges
        # Want full brightness if pred is the same for all frameskip
        maxclass = max(outdata["pred"])
        minclass = min(outdata["pred"])
        if maxclass == minclass:
            outdata["colourscale"] = 1.0
        else:
            numclasses= maxclass - minclass
            outdata["colourscale"] = outdata["pred"]/numclasses
            if(min(outdata["colourscale"]) < 0 or \
                max(outdata["colourscale"]) > 1):
                print("Colour scaling went wrong; check numbering of preciction classes")
                quit()

    if getTrackType(outdata) == "analogue":
        print "Scaling analogue values"
        scaler = MinMaxScaler()
        outdata = pd.DataFrame(scaler.fit_transform(outdata), columns = outdata.columns)

    return outdata


def getTrackType(trackdata):
    colnames = set(list(trackdata))
    colnames.add("Frame")
    if colnames.issuperset(["bb1x", "bb1y", "bb2x", "bb2y", "bb3x", "bb3y", "bb4x", "bb4y"]): 
        return "bbox"
    elif colnames == set(analogue_names):
        return "analogue"
    else:
        print "Invalid tracking data type"
        sys.exit()
        
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
    fileind = fileind + 1



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
        tracktype = getTrackType(bbox_collection[bbk])
        if tracktype  == "bbox":
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
                    cv2.line(img, p1, p2,  color =scaleColour(colours[bbk], actbb["colourscale"]), thickness = 2)
        elif tracktype == "analogue":
            if frame in bbox_collection[bbk].index:
                actrow = bbox_collection[bbk].loc[frame]
                for i in range(1,4):
                    p1 = (barchartOffsets[bbk] + 20*i, 100)
                    p2 = (barchartOffsets[bbk] + 10 + 20*i, 100 + int(100 * actrow['c' + str(i)]))
                    cv2.rectangle(img, p1, p2, 
                        colours[bbk], thickness = cv2.cv.CV_FILLED)
        else:
            print "Invalid type of tracking file"
            sys.exit()
    



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
