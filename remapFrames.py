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
    print "DEBUG - dropping some columns"
    coldel = [  "cpt" + str(x) for x in range(1,15)]

    indata = indata.drop(coldel, axis=1)
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

def kinect2webcam(originalData, framemap, offset):
    """ Convert kinect frames to webcam frames"""
    # We put the index into a normal column for the join
    # otherwise things get confusing with which index gets used
    originalData["origframe"] = originalData.index.values
    outdata = pd.merge(originalData, 
            framemap[["webcamframe", "kinectframe"]],
            left_on = "origframe",
            right_on = "kinectframe",
            sort = True
            )

    outdata["frame"] = outdata["webcamframe"]
    del outdata["kinectframe"]
    del outdata["webcamframe"]
    del outdata["origframe"]
    outdata.set_index("frame", inplace = True, verify_integrity =True)
    print "k2w converted with " + str(len(outdata)) + " rows"
    outdata.dropna(inplace = True)
    print str(len(outdata)) + " rows after deleting NAs"

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

    originalData = loadData(args.infile)
    offset = getOffset(args.frameoffsetfile, args.participant)
    framemap = loadFramemap(args.framemap, offset)

    if args.convertToKinect == True:
        print "Converting webcam to kinect"
        sys.exit("Not yet implemented")

    else:
        print "Converting kinect to webcam"
        outdata = kinect2webcam(originalData, framemap, offset)
        
    outdata.to_csv(args.outfile)

    
