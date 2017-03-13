#!/usr/bin/python

"""Convert a spot the difference attention time file to 
 the attention at each frame

 OR Extract the frame number of an event from the attention file

 TODO Allow user to choose how to handle transition periods
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import math


binarycoding = [("tablet", 1)]

def loadAttentionFile(infile, participant):
    # Load the attentionfile

    colnames = ["eventtype",
                "null",
                "attTransStarthms",
                "attTransStartss",
                "attTransStartms",
                "attTransEndhms",
                "attTransEndss",
                "attTransEndms",
                "attDurationhms",
                "attDurationss",
                "attDurationms",
                "annotation"]

    wantcols = ["eventtype", 
            "attTransStartss",
            "attTransEndss",
            "attDurationss",
            "annotation"]

    recoding = [("TV_to_+tablet", "TV_to_tablet"),
            ("Tv", "TV"),
            ("start _tablet", "start_tablet"),
            ("TV_to_tablet", "tablet"),
            ("tablet_to_TV", "TV")]
    

    if participant == "P16":
        print "Partcipant 16 found; dropping missing columns for load"
        missingcols = ["attTransStartms","attTransEndms","attDurationms"]
        colnames = list(filter(lambda x: x not in missingcols, colnames))


    attention = pd.read_table(infile, header = None,  names = colnames)
    attention = attention[wantcols]

    # Recode typos
    for oldval, newval in recoding:
       attention['annotation'] =  attention['annotation'].replace(oldval, newval)

    attention['eventtype'] = attention['eventtype'].replace("annotations", "annotation")

    # Drop missing annotations
    attention = attention[attention['annotation'].notnull()]

    attention['attTransMidss'] = attention['attTransStartss'] + (attention['attTransEndss'] - attention['attTransStartss'])/2.0

    return attention

def loadAttention(infile, participant):
    attentionAndEvents = loadAttentionFile(infile, participant)

    attention = attentionAndEvents[attentionAndEvents['eventtype'] == "attention"]
    return attention

def loadEvents(infile, participant):

    attentionAndEvents = loadAttentionFile(infile, participant)
    
    events = attentionAndEvents[attentionAndEvents['eventtype'] == "annotation"]
    return events


def getAttention(frametime, timestamps, attention):

    if len(timestamps) != len(attention): 
        print "Timestamps and attentions must be the same length"
        quit()

    if frametime < 0:
        return None

    df = pd.DataFrame({"times": timestamps, "attention": attention})
    earliertimes = df[df["times"] <= frametime]
    if len(earliertimes) == 0:
        return df["attention"][0]
    else:
        return earliertimes.iloc[-1]["attention"]

def loadExternalEvents(infile, participant):
    # Load an external event file.  This converts the free text annotations
    # indicating the start and end of each part of the experiment to 
    # a standardised form.  It also loads standardised events and times 
    # where these weren't specified in the original annotation file

    extEvents = pd.read_csv(infile)
    extEvents = extEvents[extEvents['participantCode'] == participant]

    return extEvents

def getMaxTime(attentionFile, participant):
    """ Get the maximum timestamp in a file """
    
    attention = loadAttentionFile(attentionFile, participant)
    maxtime = max(attention["attTransEndss"])
    return maxtime

def getOffset(offsetfile, participant):
    offsets = pd.read_csv(offsetfile)

    offsetrow = offsets[offsets["participantCode"] == participant]

    if len(offsetrow) != 1:
        Exception("Could not get offset for participant")


    return int(offsetrow["delta"])



def loadFramemap(mapfile, frameOffset):


    framemap = pd.read_csv(mapfile,
        header = None, names = ["kinectframe", "frametime"])

    framemap["reltime"] = framemap["frametime"] - framemap["frametime"][0]
    framemap["webcamframe"] = framemap["reltime"].apply(lambda x: 
            round(x * args.fps) + offset)
    framemap["webcamtime"] = (framemap["webcamframe"] - 1) / args.fps

    return framemap




########################################
parser = argparse.ArgumentParser(description = "Convert a ground-truth tracking file to the attention at each frame or extract the frame number of an event")

parser.add_argument("--attentionfile",
        dest = "attentionfile", type = str, required = True,
        help = "The attention file to convert")

parser.add_argument("--outputfile",
        dest = "outputfile", type = str, required = False,
        help = "The output file")


parser.add_argument("--participant",
        dest = "participant", type = str, required = True,
        help = "The participant code; Pxx")

parser.add_argument("--event",
        dest = "event", type = str, required = False,
        help = "Return the frame(s) a specified event occured in")

parser.add_argument("--externaleventfile",
        dest = "externaleventfile", type = str, required = False,
        help = "file mapping the long annotation for each participant to a standardised annotation.  May also contain additional timestamps for events that were not recorded in the attention file")

parser.add_argument("--skipfile",
        dest = "skipfile", type = str, required = False,
        help = "If specified, program will *only* output a skipfile in a format suitable for the CppMT object tracking pipeline")


parser.add_argument("--fps", dest="fps", type = int, required = False, default = 30)


parser.add_argument(
        "--framemap",
        dest = "framemap",
        type = str,
        required = False,
        help = "A list of frames and times that each video frame occured at.  Only needed if this is *different* from framelist used to encode the behaviours")


parser.add_argument(
        "--offsetfile",
        dest = "offsetfile",
        type = str,
        required = False,
        help = "A file containing the participant codes and frame offsets.  Only needed when the video we're generating an attention file for didn't start at the same time as the ones used to encode the behaviours")


args = parser.parse_args()


fps = args.fps

if args.skipfile and not args.event:
    print "An event must be specified when outputting a skipfile"
    quit()

if args.skipfile is None and args.outputfile is None:
    print "Skipfile or outfile must be specified"
    quit()


# Load data to do Kinect frame mapping
if bool(args.framemap is None) != bool(args.offsetfile is None):
    raise Exception("Must specify an offset time file AND a framemap")

if not(args.framemap is None):
    offset = getOffset(args.offsetfile, args.participant)
    framemap = loadFramemap(args.framemap, offset)


if args.event is None:
    attention = loadAttention(args.attentionfile, args.participant)

    # Get the maximum and minimum frames to encode
    maxtime = getMaxTime(args.attentionfile, args.participant)

    mintime = min(attention['attTransStartss'])
	
    frames = range(int(mintime * fps), int(maxtime * fps))
    times = [x / float(fps) for x in frames] 
	
    attention = [getAttention(x, attention["attTransMidss"], attention["annotation"]) for x in times]
	
    outdata = pd.DataFrame({"frame": frames,
	    "bbx": 150,
	    "bby": 150,
	    "bbw": 150,
	    "bbh": 150,
	    "pred": attention},
	    columns = ["frame", "bbx", "bby", "bbw", "bbh", "pred"])
    # Recode to numeric states
	
    for oldval, newval in binarycoding:
        outdata["pred"] = outdata["pred"].replace(oldval, newval)
	
    #Anything that isn't touched goes to 0
    goodvals = [x[1] for x in binarycoding]
    outdata["pred"] = np.where([w in goodvals for w in outdata["pred"]],
       outdata["pred"], 0)


else:
    if args.externaleventfile is None:
        print "Must specify the external event file"
        quit()

    print "Extracting events"
    events = loadEvents(args.attentionfile, args.participant)
    eventMapping = loadExternalEvents(args.externaleventfile, args.participant)

    eventOfInterest = eventMapping[eventMapping['event'] == args.event]
    if len(eventOfInterest) == 0:
        print "Event not found"
        quit()

    if len(eventOfInterest) > 1:
        print "Multiple events found"
        quit()

    if not math.isnan(eventOfInterest['timestamp']):
        print "Extracting event directly from time"
        eventtimeframe = eventOfInterest['timestamp']
        if len(eventtimeframe) != 1:
            print "Incorrect number of events found"
            quit()
        eventtime = eventtimeframe.iloc[0]
        eventframe = int(eventtime * fps)
    else:
        print "Extracting event via attention file"

        events  = loadEvents(args.attentionfile, args.participant)

        eventrow = events[events['annotation'] == eventOfInterest['annotation'].iloc[0]]
        if len(eventrow) != 1:
            print "Incorrect number of events found"
            quit()

        eventtime = eventrow['attTransMidss'].iloc[0]

    # Load the attention to check we've not overrun the end
    attention = loadAttention(args.attentionfile, args.participant)

    maxencodedattention  =  max(attention['attTransMidss'])
    print "Max encoded attention:", maxencodedattention
    print "Event occured at (webcam time):", eventtime
    if maxencodedattention <= eventtime:
        print "WARNING: Event is after ground truth file"
        print "Assuming last attention persisted to specified time"

    eventframe = int(eventtime * fps)

    outdata = pd.DataFrame({"frame": eventframe,
	    "bbx": 150,
	    "bby": 150,
	    "bbw": 150,
	    "bbh": 150,
	    "pred": 1},
	    columns = ["frame", "bbx", "bby", "bbw", "bbh", "pred"],
            index = [eventframe])

# Frames are 1 indexed in abc-classify:
outdata["frame"] = outdata["frame"] + 1


# This is all in webcam-frames.  If we have defined a kinect mapping
# we need to output in this
if framemap is not None:
    outdata = outdata.merge(framemap[["webcamframe", "kinectframe"]],
             left_on="frame",right_on="webcamframe", sort = True) 
    del outdata['webcamframe']
    outdata["frame"] = outdata["kinectframe"]
    del outdata["kinectframe"]
    outdata.dropna(inplace = True)


if args.skipfile is not None:
    print "Outputting skipfile"
    with open(args.skipfile, "w") as f:
        f.write(str(int(eventtime * 1000)))

else:
    outdata.to_csv(args.outputfile, index = False)


