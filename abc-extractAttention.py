#!/usr/bin/python

# Convert a spot the difference attention time file to 
# the attention at each frame

# TODO Allow user to choose how to handle transition periods

import os
import sys
import numpy as np
import pandas as pd
import argparse

fps = 30

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

    attention['attTransMidss'] = attention['attTransStartss'] + (attention['attTransEndss'] - 
            attention['attTransStartss'])

    return attention

def loadAttention(infile, participant):
    attentionAndEvents = loadAttentionFile(infile, participant)

    attention = attentionAndEvents[attentionAndEvents['eventtype'] == "attention"]
    return attention


    

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




########################################
parser = argparse.ArgumentParser(description = "Convert a ground-truth tracking file to the attention at each frame")

parser.add_argument("--attentionfile",
        dest = "attentionfile", type = str, required = True,
        help = "The attention file to convert")

parser.add_argument("--outputfile",
        dest = "outputfile", type = str, required = True,
        help = "The output file")


parser.add_argument("--participant",
        dest = "participant", type = str, required = True,
        help = "The participant code; Pxx")

args = parser.parse_args()

attention = loadAttention(args.attentionfile, args.participant)


# Get the maximum and minimum frames to encode
maxtime = max(attention['attTransEndss'])
mintime = min(attention['attTransStartss'])


frames = range(int(mintime * fps), int(maxtime * fps))
times = [x / float(fps) for x in frames] 
getAttention(-1,attention["attTransMidss"], attention["annotation"])

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

outdata.to_csv(args.outputfile, index = False)

