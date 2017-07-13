#!/usr/bin/env python


import argparse
import cv2
import numpy as np
import abcclassify.abcclassify as abcc

parser = argparse.ArgumentParser(description = "Interactively classify behaviours in a video.  For each frame enter a numeric behaviour state.  Press c to classify based on the frames classified so far.  Accuracy is evaluated with cross validation.  Can optionally use an external ground truth file for classification and/or verification.")
parser.add_argument("--videofile",
        dest = "videofile", type = str, required = False,
        help = "The input video file to classify")
parser.add_argument("--trackerfile",
        dest = "trackerfile", type = str, required = True,
        action="append",
        help = "The data from some object tracking software.  OpenFace and CppMT data will be handled appropriately.  Other data types will take the 1st colum as the frame number, and assume all other columns are required.")
parser.add_argument("--startframe", type = int, required = False,
        help = "The frame of the video to start classifcation at.  Defaults to start of video")
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False,
        help = "The end frame to run classification on.  Defaults to the end of the video")
parser.add_argument("--extgt", type = str, required = False,
        help = "Whether to use an external ground truth file(s). (currently assumed to have 6 columns; the first containing the video frame number, the sixth containing the state" )
parser.add_argument("--entergt",
        dest = "entergt", action="store_true",
        help = "Whether to interactively enter ground truth data.  For each frame enter a numeric state, c to classify or u to undo the previous frame")

parser.add_argument("--useexternalgt",
        dest = "entergt", action='store_false',
        help = "Whether to use the externally specified ground truth file for classification, instead of classifying interactivel")
parser.add_argument("--externaltrainingframes", type = int, required = False,
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



if args.extgt is None:
    print "An external ground truth file is required"
    sys.exit()

if args.videofile is not None:
    print "This version doesn't support manual coding from a video file"
    sys.exit()

participant = abcc.videotracking(framerange=(args.startframe, args.endframe))


# if args.summaryfile is not None and args.participantcode is None:
#     print "A participant code must be provided if outputting summary data"
#     sys.exit()

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
trainingframes = participant.frames

missingframecount = participant.getnumframes() - participant.getnumtrackableframes() 
if missingframecount > 0:
    print "Don't have tracking data for each frame"
    print str(missingframecount) + " frames missing"
    if missingframecount > args.maxmissing:
        print "Too many missing frames"
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
externalGT = abcc.loadExternalGroundTruth(args.extgt)

if not set(trainingframes).issubset(externalGT.index):
    print "External ground truth not provided for all frames with tracking data"
    print "Missing " + str(len(set(trainingframes) - set(externalGT.index))) + " frames of ground truth"
    quit()

quit()
trainedframescount = 0


groundtruth = []
getmulti = False

if args.entergt:
    cv2.namedWindow("Classification")
    while trainedframescount < len(trainingframes):
        thisframe = trainingframes[trainedframescount]
        if getmulti:
            img = abcc.getMultiVideoFrame(videoFile, thisframe)
            getmulti = False
        else:
            img = abcc.getVideoFrame(videoFile,thisframe)

        cv2.imshow("Classification", img)

        key =  cv2.waitKey(0)
        if(chr(key) == 'c'):
            decisionTree = abcc.runClassifier(groundtruth[:(trainedframescount)],
                    trackingData.loc[trainingframes[:(trainedframescount)]], args.forest)

            (meanAc, stdAc, nowhere, nowhere)  = abcc.getAccuracyCrossVal(decisionTree,
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])
            print("Crossval Accuracy: Mean: %0.3f, Std: %0.3f" % (meanAc, stdAc))

            if args.extgt is not None:
                print "Classification using all remaining external ground truth data:"

                predicted = decisionTree.predict(trackingData.loc[trainingframes[trainedframescount:]])
                evaluationgroundtruth = externalGT.loc[trainingframes[trainedframescount:]]

                print(metrics.classification_report(evaluationgroundtruth, predicted))
                print(metrics.confusion_matrix(evaluationgroundtruth, predicted))
                print(metrics.accuracy_score(evaluationgroundtruth, predicted))


            if args.outfile is not None:
                abcc.savePredictions(decisionTree,  trackingData,
                        trainingframes[trainedframescount:], args.outfile)
        elif(chr(key) == 'e'):

            print "Probability accuracy at least:"

            probs = abcc.getShuffledSuccessProbs(decisionTree,
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])
            print probs.mean()

            for p in [0.8, 0.9, 0.95, 0.99]:
                print "probability accuracy > " + str(p) + ": " + str(abcc.testprob(probs, p))
        elif(chr(key) == 'm'):
            getmulti = True
        elif(chr(key) == 'q'):
            if args.outfile is not None:
                print "Saving predictions"
                decisionTree = abcc.runClassifier(groundtruth[:(trainedframescount)],
                    trackingData.loc[trainingframes[:(trainedframescount)]],args.forest)
                if args.includegt:
                    abcc.savePredictions(decisionTree,  trackingData, trainingframes[trainedframescount:],
                            args.outfile,
                            groundtruthframes = trainingframes[:trainedframescount],
                            groundtruth = groundtruth[:trainedframescount])
                else:
                    abcc.savePredictions(decisionTree,  trackingData, trainingframes[trainedframescount:],
                            args.outfile)
            print "Exiting"
            sys.exit()
        elif(chr(key) == 'u'):
            print "Undoing"
            groundtruth.pop()
        elif(chr(key) == 'r'):
            cv2.destroyWindow("Classification")
            print "Playing predictions, including frames manually classified"

            decisionTree = abcc.runClassifier(groundtruth[:(trainedframescount)],
                    trackingData.loc[trainingframes[:(trainedframescount)]])

            predictions = abcc.getPredictions(decisionTree, trackingData,
                    trainingframes[trainedframescount:],
                    groundtruthframes = trainingframes[:trainedframescount],
                    groundtruth =  groundtruth)
            abcc.playbackPredictions(videoFile, predictions, startVideoFrame, endVideoFrame)
            cv2.namedWindow("Classification")
        else:
            try:
                groundtruth.append(int(chr(key)))
                if args.extgt is not None:
                    print "External GT was: " + str(int(externalGT.loc[thisframe]))
            except ValueError:
                print "Invalid behaviour state entered; must be numeric"
        trainedframescount = len(groundtruth)
        print str(trainedframescount) + " frames classified"
        print pd.Series(groundtruth).value_counts()

else:
    if args.externaltrainingframes is not None:
        trainedframescount = args.externaltrainingframes
    else:
        trainedframescount = int(raw_input("Enter training frames: "))
    groundtruthDF = externalGT.loc[trainingframes[:trainedframescount],"state"]
    groundtruth = list(groundtruthDF)

    decisionTree = abcc.runClassifier(groundtruth[:(trainedframescount)],
                trackingData.loc[trainingframes[:(trainedframescount)]], args.forest)

    (meanAc, stdAc)  = abcc.getAccuracyCrossVal(decisionTree,
                    groundtruth[:trainedframescount],
                    trackingData.loc[trainingframes[:trainedframescount]])[:2]
    print("Crossval Accuracy: Mean: %0.3f, Std: %0.3f" % (meanAc, stdAc))
    if args.noaccuracyprobs == True:
        print "Probability accuracy at least:"
        probs = abcc.getShuffledSuccessProbs(decisionTree,
        groundtruth[:trainedframescount],
        trackingData.loc[trainingframes[:trainedframescount]])
        print probs.mean()


        for p in [0.8, 0.9, 0.95, 0.99]:
            print "probability accuracy > " + str(p) + ": " + str(abcc.testprob(probs, p))


    if args.outfile is not None:
        if args.includegt:
            abcc.savePredictions(decisionTree,  trackingData, trainingframes[trainedframescount:],
                    args.outfile,
                    groundtruthframes = trainingframes[:trainedframescount],
                    groundtruth = groundtruth[:trainedframescount])
        else:
            savePredictions(decisionTree,  trackingData, trainingframes[trainedframescount:],
                    args.outfile)


# TODO - code repetition with accuracy calc
    if args.summaryfile is not None:
        print "Outputting summary"

        (xvmean, xvsd, xvlb, xvub) = abcc.getAccuracyCrossVal(decisionTree,
                        groundtruth[:trainedframescount],
                        trackingData.loc[trainingframes[:trainedframescount]])

        with(open(args.summaryfile, 'a')) as summaryfile:
                summaryfile.write(args.participantcode + "," +
                    str(trainedframescount) + "," +
                    str(startVideoFrame) + "," +
                    str(endVideoFrame) + "," +
                    str(xvmean) + "," +
                    str(xvsd) + "," + 
                    str(xvlb) + "," + 
                    str(xvub) + "," +
                    str(abcc.getAccuracy(decisionTree,
                        externalGT.loc[trainingframes[trainedframescount:],
                            "state"],
                        trackingData.loc[trainingframes[trainedframescount:]])) + "," +
                    str(missingframecount) + "," +
                    str(abcc.getF1Score(decisionTree,
                        externalGT.loc[trainingframes[trainedframescount:],
                            "state"],
                        trackingData.loc[trainingframes[trainedframescount:]]))  
                    + "\n")

if args.exporttree is not None:
    print "Saving tree"
    if type(decisionTree).__name__ == "DecisionTreeClassifier":
            tree.export_graphviz(decisionTree, out_file = args.exporttree,
                feature_names=list(trackingData.columns.values))
    elif type(decisionTree).__name__ == "RandomForestClassifier":
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


