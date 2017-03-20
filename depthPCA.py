""" Use PCA on depth data to reduce dimension ready for trying to fit
the classifier"""

import pandas as pd
import numpy as np
import loadDepth
from sklearn.decomposition import PCA, IncrementalPCA
import argparse
import glob
import re
import matplotlib.pyplot as plt


# from http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
def plot_gallery(title, images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
parser=argparse.ArgumentParser(description =
        "Experiment with using PCA to reduce dimensionality of depth data")

parser.add_argument("--infolder",
        dest = "infolder", type = str, required = True)


parser.add_argument("--frameprefix",
        dest = "frameprefix", type = str, required = True)
parser.add_argument("--framesuffix",
        dest = "framesuffix", type = str, required = False, default = ".txt.gz")
        # for participant1, part 1 is 3941 to 9324 and part 2 is 9648 to 14729
parser.add_argument("--startframe",
        dest = "startframe", type = int, required = False, default = 9648)
parser.add_argument("--endframe",
        dest = "endframe", type = int, required = False, default = 14729)
parser.add_argument("--shuffle",
        dest = "shuffle", action="store_true")
parser.add_argument("--noshuffle", 
        dest = "shuffle", action="store_false")
parser.set_defaults(shuffle=True)
parser.add_argument("--numframes", type = int, required = False)
parser.add_argument("--mindepth",
        dest = "mindepth", type = int, required = False)
parser.add_argument("--maxdepth",
        dest = "maxdepth", type = int, required = False)
parser.add_argument("--outfile",
        dest = "outfile", type = str, required = True)
parser.add_argument("--showplot", action = "store_true", required = False)



args = parser.parse_args()

if args.shuffle == False and args.numframes is not None:
        sys.exit("Cannot specify number of frames if not shuffling")

#  series with index being frame number and value being filename

framestring = args.infolder + args.frameprefix + "*" + args.framesuffix
print "using glob:" + framestring
print ("using frames between {frm} and {to}".
        format(frm=args.startframe, to=args.endframe))

frames = glob.glob(framestring)
frames.sort()

frameRegex = args.frameprefix + "(\d+)\\" + args.framesuffix
framenumbers = map(lambda x: int(re.search(frameRegex, x).group(1)),frames)
frameList = pd.Series(frames, index = framenumbers)

# Filter series to range we're interested in

frameList = frameList.loc[np.logical_and(frameList.index >= args.startframe,
        frameList.index <= args.endframe)]

if args.shuffle == True:
        print "Shuffling frames"
        frameList = frameList.sample(frac=1, replace = False)

if args.numframes is not None:
        print "Selecting " + str(args.numframes) + " at random"        
        frameList = frameList.iloc[:args.numframes]


(width, height) = loadDepth.getDepthDimensions(frameList.iloc[0])


# GLOBAL!
image_shape = (height, width)

n_components = 12 
batch_size = 200
ipca = IncrementalPCA(n_components = n_components,
        batch_size = batch_size, whiten = True)

depthFrameData = np.zeros(shape = (batch_size, width*height))

batchpos = 0
for i in range(len(frameList)):
        f = frameList.iloc[i]

        depthFrame = loadDepth.loadDepth(f, width, height)
        depthFrame = loadDepth.filterFrame(depthFrame,
                mindepth = args.mindepth, 
                maxdepth = args.maxdepth,
                recodenulls = True,
                recodevalue = 0)

        depthFrameData[batchpos,:] = depthFrame["depth"]
        batchpos = batchpos + 1
        if batchpos == batch_size: # Process batch
            print "Processing batch" + f
            ipca.partial_fit(depthFrameData)
            batchpos = 0
            print "Batch processed"
# Mop up remaining frames
if batchpos != 0:
    print "Partial batch processing"
    ipca.partial_fit(depthFrameData[:batchpos,:])
    print "Partial batch processed"


#pca = PCA(n_components=n_components, svd_solver='randomized',
#          whiten=True).fit(depthFrameData)
eigenfaces = ipca.components_.reshape((n_components, height, width))

plot_gallery("test", eigenfaces[:n_components], n_col=4, n_row=3)
#plot_gallery("frames", depthFrameData[:12], 4,3)

# We now apply the transform to frame
# Define a pandas dataframe to hold the results for everything
cptnames = map(lambda x: "cpt" + str(x), range(n_components))
pca_components = pd.DataFrame(index = frameList.index, columns = cptnames)
pca_components.index.rename("frame", inplace = True)
print "Saving"
batchpos = 0
for i in range(len(frameList)):
    f = frameList.iloc[i]
    
    depthFrame = loadDepth.loadDepth(f, width, height)
    depthFrame = loadDepth.filterFrame(depthFrame,
                mindepth = args.mindepth, 
                maxdepth = args.maxdepth,
                recodenulls = True,
                recodevalue = 0)
    depthFrameData[batchpos,:] = depthFrame["depth"]
    batchpos = batchpos + 1
    if batchpos == batch_size:
        print "Saving batch" + f
        batch_components = ipca.transform(depthFrameData)
        batch_index = frameList.index.values[((i+1)-batch_size):(i+1)]
        pca_components.loc[batch_index, cptnames] = batch_components
        batchpos = 0
if batchpos != 0:
        batch_components = ipca.transform(depthFrameData)
        batch_index = frameList.index.values[(i-batchpos+1):(i+1)]
        pca_components.loc[batch_index, cptnames] = batch_components[:batchpos,:]
        

pca_components.to_csv(args.outfile)
if args.showplot == True:
        plt.show()
#plt.plot(pca.explained_variance_)
#plt.show()
