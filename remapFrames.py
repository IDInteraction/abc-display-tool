"""Code to remap between webcam and kinect frames

David Mawdsley 21 March 2017

Functionality to take a pandas dataframe, indexed by frame and to 
convert to/from webcam and kinect frames

To make the conversion we must provide an offsetfile, which
lists the difference between webcam and kinect frames *at the staart of
the video*.  We must also provide a framelist for the participant, which maps
between the two frames of reference (since the Kinect data contains dropouts where frames have been lost)  

"""

import numpy as np
import pandas as pd
fps = 30
def loadData(inputFileName):
    indata = pd.read_csv(inputFileName, index_col = 0)
    print "Loaded input data"
    print "Using " + indata.index.name + " as index"

    return indata

def getOffset(offsetfile, participant):
    offsets = pd.read_csv(offsetfile)

    offsetrow = offsets[offsets["participantCode"] == participant]

    if len(offsetrow) != 1:
        Exception("Could not get offset for participant")


    return int(offsetrow["delta"])


def loadFramemap(mapfile, frameOffset):
    """ Load a frame map and apply the offset"""

    framemap = pd.read_csv(mapfile,
        header = None, names = ["kinectframe", "frametime"])

    framemap["reltime"] = framemap["frametime"] - framemap["frametime"][0]
    framemap["webcamframe"] = framemap["reltime"].apply(lambda x: 
            int(round(x * fps) + offset))
    framemap["webcamtime"] = (framemap["webcamframe"] - 1) / fps

    return framemap

def webcam2kinect(originalData, framemap):
    return remap(originalData, framemap, "webcamframe", "kinectframe")

def kinect2webcam(originalData, framemap):
    return remap(originalData, framemap, "kinectframe", "webcamframe")

def remap(originalData, framemap, fromvar, tovar):
    """ Convert kinect frames to webcam frames"""
    # Copy the original data before we start adding new data
    originalCopy = originalData.copy()
    # Put the index into a normal column
    originalCopy["origframe"] = originalData.index.values
    outdata = pd.merge(originalCopy, 
            framemap[[tovar, fromvar]],
            left_on = "origframe",
            right_on = fromvar,
            sort = True
            )

    outdata["frame"] = outdata[tovar]
    del outdata[fromvar]
    del outdata[tovar]
    del outdata["origframe"]
    outdata.set_index("frame", inplace = True, verify_integrity =True)
    print "converted from " + fromvar + " to " + tovar +  \
        " with " + str(len(outdata)) + " rows"

    return outdata




if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description = "Convert reference frames")

    parser.add_argument("--infile",
            dest = "infile",
            type = str,
            required = True,
            help = "The file to convert.  Frame must be in the 1st column")

    parser.add_argument("--outfile",
            dest = "outfile",
            type = str,
            required = True,
            help = "The name of the output file")

    parser.add_argument("--web2kinect",
            dest = "convertToKinect",
            action = "store_true",
            help = "Convert webcam frames to kinect frames")

    parser.add_argument("--kinect2web",
            dest = "convertToKinect",
            action = "store_false",
            help = "Convert kinect frames to webcam frames")

    parser.add_argument("--framemap",
            dest = "framemap",
            type = str,
            required = True,
            help = "A list of Kinect frames and their corresponding time")
    parser.add_argument("--test",
            dest = "test",
            action = "store_true",
            required = False)
        

    # TODO Allow offset to be specified on command line
    parser.add_argument("--frameoffsetfile",
            dest = "frameoffsetfile",
            type = str,
            required = True,
            help = "A csv file contianing participantCode, frameoffset")

    # TODO Won't be required if providing offset directly.
    # The parameter is only used in the lookup in args.frameoffsetfile 
    parser.add_argument("--participant",
            dest = "participant",
            type = str,
            required = True,
            help = "The participant code")

    parser.set_defaults(convertToKinect = False)

    args = parser.parse_args()

    inputdata = loadData(args.infile)
    
    inputrows = len(inputdata)
   
    offset = getOffset(args.frameoffsetfile, args.participant)
    framemap = loadFramemap(args.framemap, offset)

    if args.test == True:
        print "Test mode - converting w2k and back"
        midData = webcam2kinect(inputdata, framemap)
        finData = kinect2webcam(midData, framemap)

        if inputdata.equals(finData):
            print "Successfully converted w2k and back"
        else:
            if set(inputdata.columns) & set(finData.columns):
                print "Column names differ"
            if inputdata.index.name != finData.index.name:
                print "Index name differs"
            if len(inputdata) != len(finData):
                print "Number of rows differ.  This may be due to skips in kinect data"
       
        if inputdata.equals(midData):
            print "initial data = w2k data"
            print "conversion failed"
        if midData.equals(finData):
            print "Intermediate step = k2w data"
            print "conversion failed"

        quit()
    

    if args.convertToKinect == True:
        print "Converting webcam to kinect"
        outdata = webcam2kinect(inputdata, framemap)

    else:
        print "Converting kinect to webcam"
        outdata = kinect2webcam(inputdata, framemap)

    outputrows = len(outdata)

    if inputrows != outputrows:
        print "Number of rows on input and output differ"
        print "Input: " + str(inputrows)
        print "Output: " + str(outputrows)
        print args.infile  + " " + str(inputrows - outputrows) + " frames lost"
        
    outdata.to_csv(args.outfile)

    
