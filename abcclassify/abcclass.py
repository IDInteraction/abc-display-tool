#!/usr/bin/env python


import cv2
import numpy as np
import pandas as pd
import unittest
import copy
import csv


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from abc import ABCMeta, abstractmethod

def loadExternalGroundTruth(infile, ppt=None, format="checkfile"):
    import os

    if format != "checkfile":
        print "Only checkfiles implemented"
        sys.exit()
    
    if not os.path.exists(infile):
        raise ValueError("Cannot find file %s" % infile)
        
    
    with open(infile, 'rb') as csvfile:
        hasheader=csv.Sniffer().has_header(csvfile.read(2048))
        

    if hasheader:
        indata = pd.read_csv(infile, index_col=0, header = 0,
            names = ["frame", "x", "y", "w", "h", "state"])
    else:
        indata = pd.read_csv(infile, index_col=0,
            names = ["frame", "x", "y", "w", "h", "state"])
    indata.drop(["x","y","w","h"], axis=1 ,inplace = True)
    
    if ppt == None:
        print "Warning - loaded all of external ground truth file"
    else:
        indata = indata.loc[ppt.gettrackableframes()]
    nullframes = sum(indata["state"].isnull())
    if nullframes > 0:
        raise ValueError("Missing behaviour states in ground truth file / trying to set groundtruth outwith trackable frame range for " +\
             str(nullframes) + " frames")

    return indata

class groundtruthsource(object):
    __metaclass__ = ABCMeta
    """ A abstract class implementing the source of ground truth; could be from an external file
    or from interactive classification of frames """
    @abstractmethod
    def __init__(self, source=None, participant=None):
        pass

    @abstractmethod
    def getgroundtruth(self, frame):
        """ Get the groundtruth for a frame (or frames??)"""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of currently classified frames with ground truth"""
        pass
    
    @abstractmethod
    def __getitem__(self, arg):
        """ Get classification for frames (if neccessary requesting they be classified interactively)"""
        pass

    @abstractmethod
    def getpdframes(self, arg):
        """ Return a pandas dataframe for the specified frames, classifying interactively if required """
        pass    

    @abstractmethod
    def classifiableframes(self):
        """ Return the number of frames it is possible to classify """
        pass

    @abstractmethod
    def getframeswithtruth(self):
        """ Return the set of frame numbers that have ground truth"""

    def iterrows(self):
        """Iterate over the groundtruth data"""
        return self.getgroundtruth.iterrows()
        
class videogroundtruth(groundtruthsource):
    """ Ground truth as given by the user by interactively classifying a video """
    controlkeys = ['u', 's', 'q'] ## Keys that make things besides ground truth classification happen
    def __init__(self, source, participant=None):
        self.video = cv2.VideoCapture(source)
        self.groundtruth = pd.DataFrame(columns = ["frame","state"])
        self.groundtruth.set_index("frame", inplace = True)
        self.classifcationorder = [] # Used to allow undo

        self.loc = self.groundtruth.loc
        
        self.windowname = "Classification"

    def __len__(self):
        return len(self.groundtruth)

    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self.getgroundtruth(arg)
        else:
            gt = [self.__getitem__(x) for x in arg]
            return gt

    def getpdframes(self, arg, noninteractive = False):
        # classify everything in arg, then return pandas dataframe
        # getgroundtruth will classify interactively by default
        # and throw an exception if running noninteractively and 
        # it gets an unclassified frame


        for a in arg:
            self.getgroundtruth(a,noninteractive = noninteractive)

        return self.groundtruth.loc[arg]


    def getframeswithtruth(self):
        return set(self.groundtruth.index)

    # TODO repetition - put in grountruthsource
    def getstatecounts(self):
        return self.groundtruth["state"].value_counts(dropna = False)

    def getVideoFrame(self, frame, directFrame = True):
        # Return the video frame from the video
        # Pass in a  video frame number
        # This link implies setting frame directly can be problematic, but seems
        # OK on our videos
        # http://stackoverflow.com/questions/11469281/getting-individual-frames-using-cv-cap-prop-pos-frames-in-cvsetcaptureproperty

        if not directFrame: #set by time
            fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
            frameTime = 1000 * (frame-1) / fps
            self.video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, frameTime)
        else: # pull out the frame by number
            self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)

        ret, img = self.video.read()
        if ret == False:
            print "Failed to capture frame " + str(frame)
            sys.exit()

        return img

    def setgroundtruth(self, frame, state):
        """ Set the ground truth for a frame"""
        if frame in self.groundtruth.index:
            raise ValueError("Groundtruth already set for frame %d" % frame)
        newframe = pd.DataFrame({"state": [state]}, index=[frame])

        self.groundtruth = self.groundtruth.append(newframe, ignore_index = False, verify_integrity=True)
        self.classifcationorder.append(frame)

    def cleargroundtruth(self, frame):
        """ Clear the ground truth for a frame"""
        if frame not in self.groundtruth.index:
            raise ValueError("Trying to clear groundtruth for frame %d when not set" % frame)
        self.groundtruth.drop(frame, inplace=True)
        self.classifcationorder.remove(frame)

    def undolastclassification(self):
        
        if len(self.classifcationorder) < 1:
            print "No frames classified to undo"
            return
        lastframe = self.classifcationorder[-1]
        self.cleargroundtruth(lastframe)
        self.getgroundtruth(lastframe)

    def getgroundtruth(self, frame, noninteractive = False):
        """ Get the ground truth for a frame; getting the user to define it if it hasn't already been
        defined """

        if not isinstance(frame, int):
            raise ValueError("getgroundtruth accepts an int")

        if frame not in self.groundtruth.index:
            if noninteractive:
                raise KeyError("Ground truth for frame %d not set and running in non interactive mode" % frame)
            # Show the frame and get the user to classify it
            cv2.namedWindow(self.windowname)

            img = videogroundtruth.getVideoFrame(self, frame)
            cv2.imshow(self.windowname, img)
            framechar = 'x'
            while framechar not in (['0','1'] + self.controlkeys):
                key = cv2.waitKey(0) & 255 # Mask - see https://codeyarns.com/2015/01/20/how-to-use-opencv-waitkey-in-python/
                framechar = chr(key)
            print "Framechar is: %s" % framechar
            if framechar == 'u': # Undo (and redo) previous classification
                self.undolastclassification()
                self.getgroundtruth(frame) # And classify the frame we were on
            elif framechar == 'q': 
                # Quit
                pass
            elif framechar == 's':
                # Return statistics
                pass
            else: # Classify frame
                if not chr(key).isdigit():
                    raise ValueError("Unhandled classifiction key pressed")
                framestate = int(chr(key)) 
                self.setgroundtruth(frame, framestate)

        if frame not in self.groundtruth.index:
            raise ValueError("Could not get state for frame %d" % frame)

        gt = self.groundtruth.loc[frame]["state"]
        return gt       

    def classifiableframes(self):
        """ Number of classifiable frames - this will be every frame in the video """
        endframe = self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        return endframe

groundtruthsource.register(videogroundtruth)

class externalgroundtruth(groundtruthsource):
    """ Ground truth as defined in an external file """
    def __init__(self, source = None, participant = None):
        self.groundtruth = loadExternalGroundTruth(infile = source, ppt = participant)

    def getgroundtruth(self, frame, failmissing = True):
        if frame not in self.groundtruth.index:
            print "frame not found"
            if failmissing == True:
                raise ValueError("Cannot find frame %d in groundtruth" % frame)
            else:
                return None

        gt = self.groundtruth.loc[frame]["state"]
        return gt

    def getstatecounts(self):
        return self.groundtruth["state"].value_counts(dropna = False)

    def __len__(self):
        return len(self.groundtruth)

    def __getitem__(self, arg):
        return list(self.groundtruth.loc[arg]["state"])

    def getpdframes(self, arg, noninteractive=True):
        pdframe = self.groundtruth.loc[arg]
        if not isinstance(pdframe, pd.DataFrame):
            print "getpdframes should return a pandas dataframe. Quitting"
            quit()

        return pdframe

    def classifiableframes(self):
        """ All frames are classified on load for external ground truth"""
        return self.__len__(self)

    def getframeswithtruth(self):
        return set(self.groundtruth.index)

groundtruthsource.register(externalgroundtruth)


class videotracking(object):

    """ Class containing video data, tracking and classifications associated with each frame """
    def __init__(self, videofile=None, framerange=None, trackingdatafile = None):
        self.video = None
        self.frames = None # The frames we would *like* to classify (may not have tracking data for them)
        self.trackingdata = None # Contains sources of tracking information (e.g. CppMT, openface data etc.)
        self.classificationdata = None # Contains the behavioural classifications that have been set by the user
        self.classficationmethod = None # the method used to classify the frames (used to prevent us doing xvalidation on non-random classifications)
        self.numtrackingfiles = 0

        if videofile is None and framerange is None:
            print("Must supply a framerange if videofile is none")
            raise ValueError

        if videofile is not None:
            self.video = cv2.VideoCapture(videofile)

        if framerange is not None:
            if len(framerange) != 2:
                raise ValueError("Video framerange must by a tuple of length 2")
            
            if framerange[0] < 1:
                raise ValueError("First frame must be >= 1")

            if self.video is not None:
                lastframe = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                if framerange[1] > lastframe:
                    raise ValueError("Frame range must not extend beyond the end of the video")

            self.frames = range(framerange[0], framerange[1]) 
        else:
            lastframe = int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self.frames = range(1, lastframe + 1)

        # We can't store NaNs in integer numpy arrays, so we use -1 for missing    
        self.classificationdata = pd.Series(data = [-1] * len(self.frames),
            index = self.frames, dtype = np.int64)

        if trackingdatafile is not None:
            self.trackingdata = addtrackingdata(trackingdatafile)

    def split(self, framerange):
        """ Return a videotracking object containing a contiguous subset of frames """
        # TODO subset rather than copy and delete
        newvt = copy.deepcopy(self)
        # newvt.video - this will (potentially) contain more frames than we want
        if len(framerange) != 2:
            raise ValueError("framerange must be a tuple of length 2")
        if framerange[0] < min(newvt.frames) or framerange[1] > max(newvt.frames):
            raise ValueError("can only split on frames that already exist")

        # frames contains frames we'd like to classify (even though we may not have tracking data)
        # so we don't need to worry about missing frames here
        newvt.frames = range(framerange[0], framerange[1])
        newvt.trackingdata = newvt.trackingdata.loc[newvt.frames]
        newvt.classificationdata = newvt.classificationdata.loc[newvt.frames]

        return newvt

    def join(self, extravt):
        """ Join the data in extravt to self, and return this in a new object
        leaving self untouched """

        # check video is the same in both objects
        if self.video != extravt.video:
            raise ValueError("Source video file is different")
        
        newvt = copy.deepcopy(self)
        # check frame ranges don't overlap 

        if len(set(self.frames) & set(extravt.frames)) > 0:
            raise ValueError("Frames must not overlap")

        # Check same tracking data (same columns) in both parts
        if newvt.numtrackingfiles != extravt.numtrackingfiles:
            raise ValueError("Both sources must have the same number of tracking files")

        # do the join
        newvt.classificationdata = newvt.classificationdata.append(extravt.classificationdata)
        newvt.trackingdata = newvt.trackingdata.append(extravt.trackingdata)
        newvt.frames += extravt.frames

        # Sort everything in frame order
        newvt.classificationdata.sort_index(inplace=True)
        newvt.trackingdata.sort_index(inplace=True)
        newvt.frames.sort()

        # Check we have tracking data and (possibly empty) classifications for each frame
        if list(newvt.classificationdata.index) != list(newvt.trackingdata.index) or \
            list(newvt.trackingdata.index) != newvt.frames:
            raise ValueError("Error when joining objects")

        return newvt

    def trackrange(self):
        """ Return the extent of frames we (aim) to generate predictions for"""
        return self.frames 

    def gettrackableframes(self):
        """" Return frames that we have tracking data for """
        if self.numtrackingfiles == 0:
            return None
        elif len(self.trackingdata.index) == 0:
            return None
        else:
             return list(self.trackingdata.index)

    def getnumtrackableframes(self):
        return len(self.gettrackableframes())

    def getnumframes(self):
        return len(self.frames)

    def getmissingframecount(self):
        """ Get the number of frames we are missing tracking data for """
        return self.getnumframes() - self.getnumtrackableframes() 

    def loadTrackingData(self, infile, guessClean=True):
        print "Loading tracking data from: " + infile
        indata=pd.read_csv(infile, index_col=0)
        stripvals = [x.strip(' ') for x in indata.columns.values]
        indata.columns = stripvals

        if guessClean == True:
            # Remove spurious columns based on filetype
            if set(['timestamp', 'gaze_0_x']).issubset(indata.columns.values):
                print("OpenFace input detected")
                AUCols = [x.find("AU") == 0 for x in indata.columns.values]
                ControlCols = [True] * 3 + [False] * (indata.columns.values.size - 3)
                mask = [x|y for (x,y) in zip(AUCols, ControlCols)]

                indata.drop(indata.columns[mask], axis = 1, inplace = True)
            elif set(['Timestamp (ms)',
                'Active points',
                'Bounding box centre X (px)']).issubset(indata.columns.values):
                print("CppMT input detected")

                indata.index.names = ['frame']
                del indata['Timestamp (ms)']
            else:
                print("Could not recognise input format")
                print("Assuming frame is column 0")
                
                indata = pd.read_csv(infile, index_col = 0)

                if indata.index.name is None:
                    print "Using unnamed column as frame"
                else:
                    print "Using " + indata.index.name + " as frame"
                print "Using the following columns for classifier:"
                print indata.columns.values

        return indata

    def addtrackingdata(self, trackingdatafile):
        thistracking = self.loadTrackingData(trackingdatafile)
        self.numtrackingfiles += 1

        # A KeyError is thrown if we don't have tracking for all the frames in framerange
        # So we first get the subset of frames that are in the tracking data before filtering down
        trackingframes = set(thistracking.index).intersection(set(self.frames))

        if len(trackingframes) > 0:
            filteredtracking = thistracking.loc[trackingframes]
        else:
            print "Warning: no matching frames found in tracking data"
            filteredtracking = thistracking[0:0]

        if self.trackingdata is None:
            self.trackingdata = filteredtracking.copy()
        else:
            self.trackingdata = self.trackingdata.join(thistracking, how="inner", 
             rsuffix="_" + str(self.numtrackingfiles)) 
        self.trackableframes = list(self.trackingdata.index.values)


    def addframeastracking(self):
        """ Include the frame number as a tracking variable """
        self.trackingdata["frame"] = self.trackingdata.index

    def numTrackingPredictors(self):
        return len(self.getTrackingColumns())

    def numTrackingFrames(self):
        return len(self.trackingdata.index)
    
    def numTrackingFiles(self):
        return self.numtrackingfiles

    def getTrackingColumns(self):
        return list(self.trackingdata.columns)
    
    def setClassification(self, frame, state, testunset = False):
        if self.classficationmethod is None:
            raise ValueError("Must specify classification method before classifying frames")
        """ Set the behaviour classification for a frame"""
        if state == -1:
            raise ValueError("Cannot set behaviour to be -1")
        if state is None:
            state = -1

        updateseries = pd.Series(data = [state], index=[frame])

        # Test the frame we're trying to update exists
        if not set(updateseries.index).issubset(set(self.classificationdata.index)):
            raise ValueError("Attempted to update classification for a frame that does not exist" + str(updateseries))
        
        if testunset and self.classificationdata[frame] != -1:
            raise ValueError("Attempting to set an already set state, when testunset is True" + str(frame))

        self.classificationdata.update(updateseries)

    def setClassificationMethod(self, method):
        classificationmethods = ["random", "sequential"] 
        if not (method in classificationmethods):
            raise ValueError("Classification method must be one of " + str(classificationmethods))
        self.classficationmethod = method

    def getClassificationMethod(self):
        return self.classficationmethod

    def getClassification(self, frame):
        thisclassification =  self.classificationdata[frame]
        if thisclassification == -1:
            return None
        else:
            return thisclassification

    def getClassificationStates(self):
        """ Return the states that have been used to classify behaviours"""
        return set(self.getClassifiedFrames())

    def getClassifiedFrames(self):
        classifiedframes = self.classificationdata.loc[self.classificationdata != -1]
        return classifiedframes

    def getUnclassifiedFrames(self):
        unclassifiedframes = self.classificationdata.loc[self.classificationdata == -1]
        return unclassifiedframes

    def getTrackableUnclassifiedFrames(self):
        untrackinds = set(self.gettrackableframes()) - set(self.getClassifiedFrames().index)
        unclassifiedframes = self.classificationdata.loc[untrackinds]
        return unclassifiedframes


    
    def getTrackingForClassifiedFrames(self):
        tframes = self.trackingdata.loc[self.getClassifiedFrames().index]
        return tframes

    def getTrackingForUnclassifiedFrames(self):
        untrackinds = set(self.gettrackableframes()) - set( self.getClassifiedFrames().index)
        return self.classificationdata.loc[untrackinds] 

    def getTrackingForFrames(self, indices):
        return self.trackingdata.loc[indices]

    def numClassifiedFrames(self):
        numclassified = len(self.getClassifiedFrames())
        return numclassified
    
class videotrackingclassifier(object):
    """ Fit a (decision tree) classifier to a videotracking object
    
    This is a wrapper to the sklearn code, which pulls out the appropriate frames to
    run the classier on """

    def testindicies(self, frame1, frame2):
        """ Check the indicies in two dataframes are equal """
        f1i = list(frame1.index)
        f2i = list(frame2.index)

        if f1i != f2i:
            raise ValueError("Indicies do not agree")

    def __init__(self, videoTrackingObject, random_state = None):
        self.classifier = DecisionTreeClassifier(random_state = random_state)
        self.vto = videoTrackingObject
        classifiedframes = self.vto.getClassifiedFrames()
        if len(classifiedframes) < 1:
            raise ValueError("Trying to run classifier when no frames have been classified")
        trackingforclassifiedframes = self.vto.getTrackingForClassifiedFrames()

        self.testindicies(trackingforclassifiedframes, classifiedframes) 

        self.classifier.fit(trackingforclassifiedframes, classifiedframes)

    def getPredictions(self, frames):
        if not set(frames).issubset(set(self.vto.gettrackableframes())):
            raise ValueError("Trying to predict for frames without tracking data")

        dupframes = set(frames) & set(self.vto.getClassifiedFrames().index)
        if len(dupframes) > 0:
            print "****"
            print len(dupframes)
            raise ValueError("Trying to predict for frames that have already been classified:" + str(dupframes))

        trackingdata = self.vto.getTrackingForFrames(frames)

        preds = self.classifier.predict(trackingdata)
        return preds

    def getMetric(self, truth, metric):
        """ Get an accuracy-like metric for data in truth.  Truth contains the ground truth, indexed by frame
        number for the hframes we want to evaluate """
        preds = self.getPredictions(truth.index)

        metric = metric(truth, preds)
        return metric

    def getAccuracy(self, truth):
        """ Shortcut to get the accuracy, truth contains the ground truth we wish to
        predict for, and evaluate the accuracy against """
        accuracy = self.getMetric(truth, metrics.accuracy_score )
        return accuracy

    def getCrossValidatedScore(self, cv=None, random_state=None):
    
        if self.vto.getClassificationMethod() != "random":
            raise ValueError("Cross validation is only meaningful when frames have been classified at random")

        trackingdata = self.vto.getTrackingForClassifiedFrames()
        classificationdata = self.vto.getClassifiedFrames()

        self.testindicies(trackingdata, classificationdata)

        # Set cv to default for cross_val_score if unset
        if cv is None:
            cv = 3

        score = cross_val_score(self.classifier, \
            trackingdata,
            classificationdata, cv=KFold(n_splits=cv, shuffle=True, random_state=random_state))

        return score
    
    def getShuffleSplitScore(self, n_splits=None, test_size=None, random_state=None):
        
        if self.vto.getClassificationMethod() != "random":
            raise ValueError("Cross validation is only meaningful when frames have been classified at random")

        trackingdata = self.vto.getTrackingForClassifiedFrames()
        classificationdata = self.vto.getClassifiedFrames()

        self.testindicies(trackingdata, classificationdata)

        # Set parameters to ShuffleSplit default if not set
        if n_splits is None:
            n_splits = 3
        if test_size is None:
            test_size = 0.25
        
        score = cross_val_score(self.classifier, \
            trackingdata, classificationdata, cv=ShuffleSplit(n_splits = n_splits, random_state = random_state))

        return score
        

    def getClassificationMetrics(self, unclassifiedframesGT):
        """ Return a dict containing metrics and other information about the performance of the classifier.
        This contains everything, except the participantcode, that we need for the summary file""" 

        if self.vto.getClassificationMethod() == "random":
            scores = self.getCrossValidatedScore()
            xvcuts = len(scores)
            shufflesplitscores = self.getShuffleSplitScore(n_splits=100)
            sscuts = len(shufflesplitscores)
        else:
            scores = np.array(np.NaN)
            xvcuts = np.NaN
            shufflesplitscores = np.array(np.NaN)
            sscuts = np.NaN

        if unclassifiedframesGT is None: # Haven't got ground truth for unclassified frames
            groundtruthaccuracy = np.NaN
            f1 = np.NaN
        else:
            groundtruthaccuracy =  self.getAccuracy(unclassifiedframesGT)
            f1 = self.getMetric(unclassifiedframesGT, metrics.f1_score)

        summary = {"trainedframes" : len(self.vto.getClassifiedFrames()),
                   "startframe" : min(self.vto.frames),
                   "endframe": max(self.vto.frames),
                   "crossvalAccuracy" : scores.mean(),
                   "crossvalAccuracySD" : scores.std(),
                   "crossvalAccuracyLB" : np.percentile(scores,2.5),
                   "xcrossvalAccuracyUB" : np.percentile(scores,97.5),
                   "groundtruthAccuracy" : groundtruthaccuracy,
                   "missingFrames": self.vto.getmissingframecount(),
                   "f1": f1,
                   "crossvalCuts" : xvcuts,
                   "shufflesplitscores" : shufflesplitscores.mean(),
                   "shufflesplitscoresSD": shufflesplitscores.std(),
                   "shufflesplitscoresLB": np.percentile(shufflesplitscores, 2.5),
                   "shufflesplitscoresUB": np.percentile(shufflesplitscores, 97.5),
                   "shufflesplitCuts" : sscuts,
                   }

        return summary
