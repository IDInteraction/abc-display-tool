import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from sklearn import mixture
import glob
import loadDepth
import re
import math
import argparse
from itertools import chain

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

    return Polygon(poly)


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

def filterFrame(inframe, mindepth = None, maxdepth = None, polygon = None):
    """Filter a frame by depth and / or masking polygon"""

    filterframe = inframe

    #TODO could do these in one go if min and max defined
    if mindepth is not None:
        filterframe = filterframe[mindepth <= filterframe["depth"]]

    if maxdepth is not None:
        filterframe = filterframe[maxdepth >= filterframe["depth"]]
    if polygon is not None:
        print "Masking polygon"
        filterframe["inbox"] = filterframe.apply(lambda i: polygon.contains(Point( (i.x, i.y))), axis = 1)
        filterframe = filterframe[filterframe["inbox"] == True]
        print "Polygon masked"

    return filterframe

def depthImage(filteredframe, width, height):
    """ Take a filtered frame (i.e. not every (x,y) is defined) and
    return a frame with all (x,y) defined, and "None" where missing data """

    fullframe = pd.DataFrame()
    fullframe["x"] = range(width) * height
    fullframe["y"] = [item for item in range(height) for i in range(width)]
    fullframe = fullframe.merge(filteredframe, how="left", on=["x","y"])

    return(fullframe)


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
parser.add_argument("--bbox"
        , dest = "bbox", type=str, required = False)

args = parser.parse_args()

framestring = args.infolder + args.frameprefix + "*" + args.framesuffix
print "using glob:" + framestring

frames = glob.glob(framestring)
frames.sort()

print "Using frame ", frames[0], " as reference"
(width, height) = loadDepth.getDepthDimensions(frames[0])

fitframe = loadDepth.loadDepth(frames[0], width, height)

# Default depths (defined in args above) determined from Shiny app;
#covers as wide a range as possible  while capturing participant and table
mindepth = args.mindepth

maxdepth = args.maxdepth

print "Using depths between " + str(mindepth) + " and " + str(maxdepth)

#print fitframe
#print genPolygon(bboxdata.iloc[0])
#quit()

x = np.linspace(mindepth, maxdepth).reshape(-1,1)

polygon = None
if args.bbox is not None:
    bboxdata = readBoundingBox(args.bbox)
    # get polygon for first frame
    print "WARNING - JUST USING FIRST BBOX - FRAMES NOT IN SYNC"
    print "JUST FOR DEVELOPMENT"
    polygon = genPolygon(bboxdata.iloc[0])


filterdepth = filterFrame(fitframe, 
        mindepth = mindepth,
        maxdepth = maxdepth,
        polygon = polygon)



if args.numcomponents == 0:
    print "Setting num components automatically via BIC"
    components = np.arange(1,8)
    models = [None for i in range(len(components))]

    for c in range(len(components)):
        models[c] = mixture.GaussianMixture(n_components = components[c])
        models[c].fit(filterdepth["depth"].reshape(-1,1))
        
    BIC = [m.bic(filterdepth["depth"].reshape(-1,1)) for m in models]

    n_components = components[np.argmin(BIC)]
else:
    print "Setting num components from command line"
    n_components = args.numcomponents


print "Fitting all frames with ", n_components, " components"
results = []
maxheight = 1 # for plotting; is set to max value of 1st frame
fig = plt.figure()

frameRegex = args.frameprefix + "(\d+)\\" + args.framesuffix
print "Matching frames using:"
print frameRegex

for f in frames:
    
    framenum = int(re.search(frameRegex, f).group(1))
    
    print framenum
    polygon = None
    if bboxdata is not None:
        polygon = genPolygon(bboxdata.iloc[int(framenum)])
    print "DEBUG ONLY"
    print polygon
    print polygon.area
    print list(polygon.exterior.coords)
    depthdata = loadDepth.loadDepth(f, width, height)
    filterdepth = filterFrame(depthdata,
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
        if framenum == 1:
            maxheight = max(pdf)
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
