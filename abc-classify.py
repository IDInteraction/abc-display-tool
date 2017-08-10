#!/usr/bin/env python


import argparse
import cv2
import numpy as np
import abcclassify as abcc
import csv
import os.path
from sklearn import tree
import sys

parser = argparse.ArgumentParser(description = "Classify behaviours in a video.  For each frame enter a numeric behaviour state.  Press c to classify based on the frames classified so far.  Accuracy is evaluated with cross validation.  Can optionally use an external ground truth file for classification and/or verification.")
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
parser.add_argument("--includeframefortracking", action="store_true",
        help = "Include frame number as a tracking variable")
parser.add_argument("--externaltrainingframes", type = int, required = True,
        help = "The number of frames to use for training and local classification if using an external ground truth file.")

parser.set_defaults(entergt=True)

parser.add_argument("--shuffle", dest="shuffle", action="store_true",
        help = "Whether to classify frames in a random order (default)")
parser.add_argument("--noshuffle", dest="shuffle", action="store_false",
        help = "Whether to classify frames in the order they appear in the video")
parser.set_defaults(shuffle=True)

parser.add_argument("--targetted", dest="targetted", action="store_true")
parser.add_argument("--windowsize", type=int)
parser.add_argument("--advancesize", type=int)
parser.add_argument("--batchsize", type=int)


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
# This needs renaming to --configstring at some point
parser.add_argument("--participantcode",
        dest = "participantcode", type = str, required = False,
        help = "The configuration string (indicating the options used) to output to the summaryfile")
parser.add_argument("--part",
        dest = "part", type = int, required = False,
        help = "The experiment part. This will be included in the generated config string. Only required if not providing a configuration string.")
parser.add_argument("--pcode",
        dest = "pcode", type = str, required = False,
        help = "The participant code to be included in the config string")


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

if args.targetted and (args.windowsize is None or args.advancesize is None or args.batchsize is None):
        parser.error("If --targetted is true, --windowsize, --advancesize and --batchsize must be specified")

if args.targetted and args.shuffle is False:
        parser.error("Must use --shuffle if doing targetted training")

if args.videofile is not None:
    parser.error("This version doesn't support manual coding from a video file")

if args.forest != 1:
    print "Random forests not currently supported"
    quit()


# if (not args.entergt) and args.extgt is None:
#     print "If not entering ground-truth from video frames, external ground truth must be provided"
#     sys.exit()


# if args.summaryfile is not None and args.entergt:
#     print "Must use external ground truth file if outputting summary stats"
#     sys.exit()

if args.summaryfile is not None and args.participantcode is None and (args.part is None or args.pcode is None):
        parser.error("Must specify [experiment] part and participantcode if outputting a summary file (or provide a manual config string")


participant = abcc.videotracking(framerange=(args.startframe, args.endframe))

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


if args.includeframefortracking:
        print "Including frame # as tracking variable"
        participant.addframeastracking()

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

if args.targetted is True:
        print "Targetted training"
        (numbatches, remainder) = divmod(args.externaltrainingframes, args.batchsize)
        batches = [args.batchsize] * numbatches 
        if remainder != 0:
                print "Warning - training frames don't divide equally"
                batches.append(remainder)
        print "training in the following batches (first at random)" 
        print batches
        abcc.trainRegion(participant, externalGT.loc[trainingframes[:batches.pop(0)]])
        for bs in batches:
                batchresults = abcc.calcWindowedAccuracy(participant, args.windowsize, args.advancesize)
                minstartframe = batchresults["startframe"].iloc[np.nanargmin(batchresults["mean"])]
                minendframe = batchresults["endframe"].iloc[np.nanargmin(batchresults["mean"])]
                # Get frames we to classify in this range

                trainrange = range(minstartframe, minendframe)
                if max(trainrange) > max(participant.gettrackableframes()):
                        raise IndexError("Warning - training range extends beyond trackable region")
                
                # Remove frames we've already classified from the range of interest
                trainingframes = list(set(trainrange) - \
                                set(participant.getClassifiedFrames().index))
                if len(trainingframes) < args.batchsize:
                        print "Warning - classifying all frames at " + str(minstartframe)
                # Randomise and select the appropriate number of frames for the 
                # targetted training
                np.random.shuffle(trainingframes)
                trainingframes = trainingframes[:bs]
                traindata = externalGT.loc[trainingframes]
                abcc.trainRegion(participant, traindata)

else:
        for f in trainingframes[:args.externaltrainingframes]:
                participant.setClassification(f, externalGT.loc[f]["state"], testunset = True)
        print str(participant.numClassifiedFrames()) + " frames classified using ground truth"

vtc = abcc.videotrackingclassifier(participant)  # TODO - RANDOM STATE
unclassifiedframeGT = externalGT.loc[participant.getTrackableUnclassifiedFrames().index]

print str(participant.numClassifiedFrames()) + " frames classified"

metrics = vtc.getClassificationMetrics(unclassifiedframeGT)
if args.summaryfile is not None:
        print "Outputting summary file"
        if args.participantcode is None:
                configstring = ""
                # generate config string
                configstring += ("a:" + str(args.part) + "::")
                configstring += ("f:" + str(args.externaltrainingframes) + "::")
                configstring += ("p:" + args.pcode + "::") 
                configstring += "s:" 
                if args.targetted:
                        configstring += ("targetted" + str(args.windowsize) + "x" + str(args.advancesize) + "x" + str(args.batchsize))
                elif not args.shuffle:
                        configstring += "no"
                elif args.shuffle:
                        configstring += ""
                else:
                        ValueError("Could not determine training mode")
                configstring += "::"
                configstring += "i:"
                if args.includeframefortracking:
                        configstring += "incframe"
                else:
                        configstring += "excframe"
                
        else:
                configstring = args.participantcode

        metrics["configuration"] = configstring 

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
                    "crossvalCuts",
                    "shufflesplitscores",
                    "shufflesplitscoresSD",
                    "shufflesplitscoresLB",
                    "shufflesplitscoresUB",
                    "shufflesplitCuts", ]
        
        # Output header if a new file
        if not os.path.isfile(args.summaryfile):
                with open(args.summaryfile, "w") as csvfile:
                        writer = csv.DictWriter(csvfile,  fieldnames = fieldorder)
                        writer.writeheader()

        with open(args.summaryfile, "a") as csvfile:
                writer = csv.DictWriter(csvfile,  fieldnames = fieldorder)
                writer.writerow(metrics)

        print metrics

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


