import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.path as mpltPath
from sklearn import mixture
import glob
import loadDepth
import re
import math
import argparse
from itertools import chain
import pickle
import os

GRAY =   '#999999'
def genPolygon(inrow):
    numcoord = len(inrow)

    if numcoord % 2 != 0:
        Exception("Even number of values needed to make polygon")
    poly = []
    for i in range(0,numcoord,2):
        p1 = inrow[i]
        if(i == numcoord - 1):
            p2 = inrow[0]
        else:
            p2 = inrow[i+1]

        poly.append((p1, p2))

    return poly


def readBoundingBox(infile):
    """ Read bounding box information in from a file """
    bboxdata = pd.read_csv(infile, header = 0)

    if "Frame" in bboxdata.columns.values:
        print "Using 'Frame' as index col"
        bboxdata.set_index("Frame", verify_integrity = True, inplace = True)
    elif "frame" in bboxdata.columns.values:
        print "using 'frame' as index col"
        bbox.set_index("frame", verify_integrity = True, inplace = True)
    else:
        print "Could not determine frame.  Aborting"
        quit()

    if "Active points" in bboxdata.columns.values:
        print "Detected data as cppMT"
        vertexColumns = [ 'Bounding box vertex 1 X (px)',
                 'Bounding box vertex 1 Y (px)',
                 'Bounding box vertex 2 X (px)',
                 'Bounding box vertex 2 Y (px)',
                 'Bounding box vertex 3 X (px)',
                 'Bounding box vertex 3 Y (px)',
                 'Bounding box vertex 4 X (px)',
                 'Bounding box vertex 4 Y (px)']
    else:
        print "Could not determine bbox format"
        quit()


    bboxdata = bboxdata[vertexColumns]

    # Standardise names
    vertices = range(1,5)
    standardNames = zip(map(lambda x: "v" + str(x) + "x", vertices ), 
        map(lambda y: "v" + str(y) + "y", vertices ))
    standardNames = list(chain(*standardNames))

    bboxdata.columns = standardNames
    bboxdata.index.rename("frame", inplace= True)

    return bboxdata


def depthImage(filteredframe, width, height):
    """ Take a filtered frame (i.e. not every (x,y) is defined) and
    return a frame with all (x,y) defined, and "None" where missing data """

    fullframe = pd.DataFrame()
    fullframe["x"] = range(width) * height
    fullframe["y"] = [item for item in range(height) for i in range(width)]
    fullframe = fullframe.merge(filteredframe, how="left", on=["x","y"])

    return(fullframe)

def processFrame(frame, mindepth, maxdepth, polygon, components):
    """ given a loaded frame, filter by depth (and optionally bbox), 
    and then fit a mixture model with the specified number of components"""
    
    filterdepth = loadDepth.filterFrame(frame, 
            mindepth = mindepth,
            maxdepth = maxdepth,
            polygon = polygon)

    model = mixture.GaussianMixture(n_components = components)
    model.fit(filterdepth["depth"].reshape(-1,1))

    return model

def most_common(lst):
    """http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list"""
    return max(set(lst), key=lst.count)
#########################################

np.random.seed(0)

parser = argparse.ArgumentParser(description = "Load depth data, and fit mixture model")

parser.add_argument("--infolder",
        dest = "infolder", type = str, required = True)
parser.add_argument("--outfile",
        dest = "outfile", type = str, required = True)
parser.add_argument("--mindepth",
        dest = "mindepth", type = int, required = False, default = 810)
parser.add_argument("--maxdepth",
        dest = "maxdepth", type = int, required = False, default =  1710)
parser.add_argument("--frameprefix",
        dest = "frameprefix", type = str, required = True)
parser.add_argument("--framesuffix",
        dest = "framesuffix", type = str, required = False, default = ".txt.gz")
parser.add_argument("--outframefile",
        dest = "outframefile", type= str, required = False)
parser.add_argument("--numcomponents"
        , dest = "numcomponents", type=int, required = False, default = 0)
parser.add_argument("--componentsample", type = int,
        required = False,
        help = "The number of frames to sample to determine the number of components")
parser.add_argument("--maxcomponents", type = int, required = False,
        default = 8,
        help = "The maximum number of components to try and fit when estimating optimum number via BIC")

parser.add_argument("--bbox"
        , dest = "bbox", type=str, required = False)
parser.add_argument("--startframe",
        dest = "startframe", type = int, required = True)
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = True)
parser.add_argument("--pickle", action='store_true',
    help = "Store the arguments as a pickle file") 

args = parser.parse_args()
if args.componentsample is not None and args.numcomponents != 0:
    sys.exit("Cannot specify number of components if determining number of components from frame sample")

if args.pickle:    
    pickleName =  os.path.splitext(args.outfile)[0] + ".pickle"
    with open(pickleName, "wb") as fileHandle:
        pickle.dump(args.__dict__, fileHandle, protocol = 0)

framerange = range(args.startframe, args.endframe + 1)
#  series with index being frame number and value being filename

framestring = args.infolder + args.frameprefix + "*" + args.framesuffix
print "using glob:" + framestring
frames = glob.glob(framestring)
frames.sort()

frameRegex = args.frameprefix + "(\d+)\\" + args.framesuffix
framenumbers = map(lambda x: int(re.search(frameRegex, x).group(1)),frames)
frameList = pd.Series(frames, index = framenumbers)

# Filter series to range we're interested in
frameList = frameList.loc[np.logical_and(frameList.index >= args.startframe,
        frameList.index <= args.endframe)]

del frames # so we don't use by accident
if len(frameList) == 0:
    sys.exit("Couldn't find any matching frames")
(width, height) = loadDepth.getDepthDimensions(frameList.iloc[0])


# Default depths (defined in args above) determined from Shiny app;
#covers as wide a range as possible  while capturing participant and table
mindepth = args.mindepth

maxdepth = args.maxdepth

print "Using depths between " + str(mindepth) + " and " + str(maxdepth)

x = np.linspace(mindepth, maxdepth).reshape(-1,1)

polygon = None
if args.bbox is not None:
    # Read in the bounding boxes for all frames we have tracking data for
    bboxdata = readBoundingBox(args.bbox)
    # check we have tracking data for all these frames 
    if not set(frameList.index).issubset(bboxdata.index):
        sys.exit("Don't have tracking frames for all frames of interest")



componentVotes = [] # List of "votes" for optimum number of components in each frame
if args.numcomponents == 0:
    if args.componentsample is None:
        print "Setting num components automatically via BIC from 1st frame"
        componentFrames = frameList[0:1]
    
    else:
        if args.componentsample < 1:
            sys.exit("Must sample 1 or more frames")
        # Get a random sample of frames
        componentFrames = frameList.sample(frac=1)[:args.componentsample]

    for i, componentFrame in componentFrames.iteritems():
        print "Using frame " + str(i) + " to estimate number of components" 
        fitframe = loadDepth.loadDepth(componentFrame, width, height)

        if args.bbox is not None:
            polygon = genPolygon(bboxdata.iloc[i])
        else:
            polygon = None
        
        components = np.arange(1,args.maxcomponents)
        models = [None for i in range(len(components))]

        for idx, c in enumerate(components):
            models[idx] = processFrame(fitframe, mindepth, maxdepth, polygon, c)       
        filterdepth = loadDepth.filterFrame(fitframe, 
                mindepth = mindepth,
                maxdepth = maxdepth,
                polygon = polygon)
        BIC= [m.bic(filterdepth["depth"].reshape(-1,1)) for m in models]

        # Plotting the BICS we see several local minima; the first of these
        # looks to provide a reasonable fit to the histogram, (hopefully) without
        # overfitting 
        # Though BIC *should* take account of complexity and fit anyway

        # TODO Stop as soon as BIC starts increasing, rather than calculating
        # for all components
        bestBIC = float("inf")

        for idx, thisBIC in enumerate(BIC):
            if thisBIC < bestBIC:
                bestBIC = thisBIC
                theseComponents = components[idx]
            else:
                break # BIC going up again

        componentVotes.append(theseComponents)
#        componentVotes.append(components[np.argmin(BIC)])
    print componentVotes
    n_components = most_common(componentVotes)
    print "Components set to: " + str(n_components)
else:
    print "Setting num components from command line"
    n_components = args.numcomponents

print "Fitting all frames with ", n_components, " components"
results = []
maxheight = 1 # for plotting; is set to max value of 1st frame
heightset = False
fig = plt.figure()


for i in range(len(frameList)):
    f = frameList.iloc[i]
    
    framenum = frameList.index[i] 
    
    print f, framenum
    polygon = None
    try:
        bboxdata
    except NameError:
        pass
    else:
        polygon = genPolygon(bboxdata.loc[framenum])

    
    depthdata = loadDepth.loadDepth(f, width, height)
    filterdepth = loadDepth.filterFrame(depthdata,
            mindepth = mindepth,
            maxdepth = maxdepth,
            polygon = polygon)
    model = mixture.GaussianMixture(n_components = n_components)
    model.fit(filterdepth["depth"].reshape(-1,1))
    

    means = [item for sublist in model.means_.tolist() for item  in sublist]  
    covars = [item[0] for sublist in model.covariances_.tolist() for item  in sublist]  
    weights = model.weights_.tolist()

    componentOrder =  np.argsort(means)

    meansort = [means[i] for i in componentOrder]
    covarsort = [covars[i] for i in componentOrder]
    weightsort = [weights[i] for i in componentOrder]


    results.append([framenum] + meansort + covarsort + weightsort)

    if args.outframefile != None:
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        logprob = model.score_samples(x)
        pdf = np.exp(logprob)
        if not heightset:
            maxheight = max(pdf)
            heightset = True
        ax.hist(filterdepth["depth"].reshape(-1,1),200, normed=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf)
        for i in range(len(meansort)):
            plt.plot(x, mlab.normpdf(x, meansort[i], math.sqrt(covarsort[i]))*weightsort[i])
        plt.ylim([0, maxheight])
        ax = fig.add_subplot(1,2,2)

        depthdata = depthImage(filterdepth, width, height)

        grid = depthdata["depth"].reshape(height, width) 

        plt.imshow(grid, extent=(0, width, 0, height)) 
        plt.savefig(args.outframefile + str(framenum).zfill(6) + ".png")
        plt.close(fig)


header = ["frame"] + ["mean" + str(x) for x in range(n_components)] + ["variance" + str(x) for x in range(n_components)] + ["weight" + str(x) for x in range(n_components)]


df = pd.DataFrame(results, columns = header)
df = df.set_index("frame", verify_integrity = True)

df.to_csv(args.outfile)






#model_best = models[np.argmin(BIC)]
#x = np.linspace(mindepth, maxdepth).reshape(-1,1)
#logprob = model_best.score_samples(x)
#pdf = np.exp(logprob)
#
#fig = plt.figure()
#
#
#ax = fig.add_subplot(1,2,1)
#
#ax.plot(components, BIC)
#
#ax = fig.add_subplot(2,2,1)
#
#ax.hist(filterdepth["depth"].reshape(-1,1),200, normed=True, histtype='stepfilled', alpha=0.4)
#ax.plot(x, pdf)
#
#print "Means"
#print model_best.means_
#print "Covariance"
#print model_best.covariances_
#print "Weight"
#print model_best.weights_
#
#plt.show()
