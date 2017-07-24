#!/usr/bin/env python


import argparse
import cv2
import numpy as np
import abcclassify as abcc
import csv
import os.path
from sklearn import tree
import sys

parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video.  For each frame enter a numeric behaviour state.  Press c to classify based on the frames classified so far.  Accuracy is evaluated with cross validation.  Can optionally use an external ground truth file for classification and/or verification.")
parser.add_argument("--videofile",
        dest = "videofile", type = str, required = False,
        help = "The input video file to classify. Depreciated.")
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True,
        action="append",
        help = "The data from some object tracking software.  OpenFace and CppMT data will be handled appropriately.  Other data types will take the 1st colum as the frame number, and assume all other columns are required.")
parser.add_argument("--startframe", type = int, required = False,
        help = "The frame of the video to start classifcation at.  Defaults to start of video")
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False,
        help = "The end frame to run classification on.  Defaults to the end of the video")
parser.add_argument("--extgt", type = str, required = True,
        help = "Whether to use an external ground truth file(s). (currently assumed to have 6 columns; the first containing the video frame number, the sixth containing the state" )

parser.add_argument("--useexternalgt",
        dest = "entergt", action='store_false',
        help = "Whether to use the externally specified ground truth file for classification, instead of classifying interactivel")
parser.add_argument("--externaltrainingframes", type = int, required = True,
        help = "The number of frames to use for training and local classification if using an external ground truth file.  Will be prompted for if not specified")

parser.set_defaults(entergt=True)

parser.add_argument("--shuffle", dest="shuffle", action="store_true",
        help = "Whether to classify frames in a random order (default)")
parser.add_argument("--noshuffle", dest="shuffle", action="store_false",
        help = "Whether to classify frames in the order they appear in the video")
parser.set_defaults(shuffle=True)

parser.add_argument("--outfile",
        dest="outfile", type = str, required = False,
        help = "The filename to output classifier performance on the data that haven't been used to construct the classifier")

parser.add_argument("--rngstate",
        dest="rngstate", type=str, required = False,
        help = "Load a random number state file, if it exists.  If it does not exist, write the closing rng state to the file.")

parser.add_argument("--chainrngstate",
        dest="chainrngstate", action="store_true", required = False,
        help = "Whether to chain the random number state; i.e. save the state at the end of the run to the file specified in rngstate.  ")

parser.add_argument("--nochainrngstate",
        dest="chainrngstate", action="store_false", required = False,
        help = "Don't chain the RNG state.  Load the state from the file specified in rngstate if it exists.  Otherwise, seed randomly and save the seed to the file specified in rngstate")
parser.set_defaults(chainrngstate=True)

parser.add_argument("--summaryfile",
        dest = "summaryfile", type = str, required = False,
        help = "A text file to append summary information from the run to.  Currently records participantCode, trainingframes, startframe, endframe, accuracy (local), accuracy (external)")
parser.add_argument("--participantcode",
        dest = "participantcode", type = str, required = False,
        help = "The participant code to output in the summaryfile")

parser.add_argument("--noaccuracyprobs",
        dest = "noaccuracyprobs", action="store_true",
        help = "Whether to output the (slow to calculate) p accuracy > x statistics")
parser.set_defaults(noaccuracyprobs = False)

parser.add_argument("--includegt",
        dest = "includegt", action = "store_true",
        help = "Whether to include ground truth frames when outputting predictions")
parser.set_defaults(includegt=False)

parser.add_argument("--exporttree",
        dest = "exporttree", type = str, required = False)

parser.add_argument("--maxmissing",
        dest = "maxmissing", type = int,
        required = False, default = 0,
        help = "The maximum number of missing frames to allow.  Useful if using data derived from the Kinect, where frames are occasionally dropped")

parser.add_argument("--forest",
        dest = "forest", type = int,
        required = False, default = 1,
        help = "Number of trees in the forest")

args = parser.parse_args()

if args.videofile is not None:
    print "This version doesn't support manual coding from a video file"
    sys.exit()

if args.forest != 1:
    print "Random forests not currently supported"

participant = abcc.videotracking(framerange=(args.startframe, args.endframe))


if args.summaryfile is not None and args.participantcode is None:
        print "A participant code must be provided if outputting summary data"
        sys.exit()

# if (not args.entergt) and args.extgt is None:
#     print "If not entering ground-truth from video frames, external ground truth must be provided"
#     sys.exit()


# if args.summaryfile is not None and args.entergt:
#     print "Must use external ground truth file if outputting summary stats"
#     sys.exit()

# TODO handle rng seed saving
if args.rngstate is not None:
    print "warning rng state not implemented"
    if os.path.isfile(args.rngstate):
        print "Saved random state exits; loading"
        with open(args.rngstate, 'rb') as input:
            state = pickle.load(input)
        np.random.set_state(state)
    else:
        print "Random state file not found - setting"
        np.random.seed()
        state = np.random.get_state()
        if not args.chainrngstate:
            print "Saving initial random seed"
            with open(args.rngstate, 'wb') as output:
                pickle.dump(state, output, pickle.HIGHEST_PROTOCOL)



print "Tracking between " + str(min(participant.frames)) + " and " + str(max(participant.frames))

while len(args.trackerfile) > 0:
    participant.addtrackingdata(args.trackerfile.pop())

# print "The following columns are available to the classifier"
# print participant.getTrackingColumns() 

# We handle the training period by shuffling all the frames in the video
# We can then work our way through the list as required, to avoid re-drawing the sample
# and risking classifying the same frame twice etc.
trainingframes = participant.gettrackableframes() 

missingframecount = participant.getmissingframecount()
if missingframecount > 0:
    print "Don't have tracking data for each frame"
    print str(missingframecount) + " frames missing"
    if missingframecount > args.maxmissing:
        print "Too many missing frames:"
        # Print this as a numpy array to abbreviate if there are lots
        print np.array(list(set(trainingframes) - set(trackingData.index)))
        quit()

if args.shuffle:
    print "Randomising frames"
    # Must use np.random.shuffle for reproducibility, not shuffle since we have *only* set
    # numpy's random seed
    np.random.shuffle(trainingframes)
    participant.setClassificationMethod("random")
else:
    print "Using sequential frames"
    participant.setClassificationMethod("sequential")

print "Loading external ground-truth file"
externalGT = abcc.loadExternalGroundTruth(args.extgt, participant)
print externalGT["state"].value_counts(dropna = False)
print str(len(externalGT)) + " frames of ground truth loaded"
print str(participant.getnumtrackableframes()) + " trackable frames"

if not set(trainingframes).issubset(externalGT.index):
    print "External ground truth not provided for all frames with tracking data"
    print "Missing " + str(len(set(trainingframes) - set(externalGT.index))) + " frames of ground truth"
    quit()

# Classify the (random or sequential) frames we've decided to classify, using the external ground truth

for f in trainingframes[:args.externaltrainingframes]:
    participant.setClassification(f, externalGT.loc[f]["state"], testunset = True)
    

print str(participant.numClassifiedFrames()) + " frames classified using ground truth"

vtc = abcc.videotrackingclassifier(participant)  # TODO - RANDOM STATE

unclassifiedframeGT = externalGT.loc[trainingframes[args.externaltrainingframes:]]
metrics = vtc.getClassificationMetrics(unclassifiedframeGT)

print str(participant.numClassifiedFrames()) + " frames classified"

if args.summaryfile is not None:
        print "Outputting summary file"
        metrics["configuration"] = args.participantcode

        fieldorder = ["configuration",
                    "trainedframes" ,
                    "startframe" ,
                    "endframe",
                    "crossvalAccuracy" ,
                    "crossvalAccuracySD" ,
                    "crossvalAccuracyLB" ,
                    "xcrossvalAccuracyUB" ,
                    "groundtruthAccuracy" ,
                    "missingFrames",
                    "f1",
                    "crossvalCuts" ]
        
        # Output header if a new file
        if not os.path.isfile(args.summaryfile):
                with open(args.summaryfile, "w") as csvfile:
                        writer = csv.DictWriter(csvfile,  fieldnames = fieldorder)
                        writer.writeheader()

        with open(args.summaryfile, "a") as csvfile:
                writer = csv.DictWriter(csvfile,  fieldnames = fieldorder)
                writer.writerow(metrics)

if args.exporttree is not None:
    print "Saving tree"
    if type(vtc.classifier).__name__ == "DecisionTreeClassifier":
            tree.export_graphviz(vtc.classifier, out_file = args.exporttree,
                feature_names=list(vtc.vto.getTrackingColumns()))
    elif type(decisionTree).__name__ == "RandomForestClassifier":
        print "Random forests not yet implemented"
        quit()
        treenum = 0
        for dtree in decisionTree.estimators_:
            tree.export_graphviz(dtree,
                    out_file = str(treenum) + args.exporttree,
                    feature_names=list(trackingData.columns.values))
            treenum += 1
    else:
        sys.exit("Invalid classifier object")


if args.rngstate is not None:
    if args.chainrngstate:
        print "Saving final RNG state"
        state = np.random.get_state()
        with open(args.rngstate, 'wb') as output:
            pickle.dump(state, output, pickle.HIGHEST_PROTOCOL)
    else:
        print "Not chaining RNG state"


