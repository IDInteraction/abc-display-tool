import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import mixture
import glob
import loadDepth
import re
import math
import argparse


np.random.seed(0)

parser = argparse.ArgumentParser(description = "Load depth data, and fit mixture model")

parser.add_argument("--infolder", dest = "infolder", type = str, required = True)
parser.add_argument("--outfile", dest = "outfile", type = str, required = True)
parser.add_argument("--mindepth", dest = "mindepth", type = int, required = False, default = 810)
parser.add_argument("--maxdepth", dest = "maxdepth", type = int, required = False, default =  1710)
parser.add_argument("--frameprefix", dest = "frameprefix", type = str, required = True)
parser.add_argument("--framesuffix", dest = "framesuffix", type = str, required = False, default = ".txt.gz")
parser.add_argument("--outframefile", dest = "outframefile", type= str, required = False)


args = parser.parse_args()

framestring = args.infolder + args.frameprefix + "*" + args.framesuffix
print "using glob:" + framestring

frames = glob.glob(framestring)
frames.sort()

print "Using frame ", frames[0], " as reference"
(width, height) = loadDepth.getDepthDimensions(frames[0])

fitframe = loadDepth.loadDepth(frames[0], width, height)

# Depths determined from Shiny app; covers as wide a range as possible
# while capturing participant and table
mindepth = args.mindepth

maxdepth = args.maxdepth

print "Using depths between " + str(mindepth) + " and " + str(maxdepth)

filterdepth = fitframe[np.logical_and(mindepth <= fitframe["depth"] , maxdepth >= fitframe["depth"]) ]

components = np.arange(1,8)

x = np.linspace(mindepth, maxdepth).reshape(-1,1)

models = [None for i in range(len(components))]

for c in range(len(components)):
    models[c] = mixture.GaussianMixture(n_components = components[c])
    models[c].fit(filterdepth["depth"].reshape(-1,1))
    
BIC = [m.bic(filterdepth["depth"].reshape(-1,1)) for m in models]

n_components = components[np.argmin(BIC)]

print "Fitting all frames with ", n_components, " components"
results = []
maxheight = 1 # for plotting; is set to max value of 1st frame
fig = plt.figure()

for f in frames:
    
    framenum = int(re.search("(\d+)\.txt$", f).group(1))
    print framenum

    depthdata = loadDepth.loadDepth(f, width, height)
    filterdepth = depthdata[np.logical_and(mindepth <= depthdata["depth"], maxdepth >= depthdata["depth"])]
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
        depthdata.loc[np.invert( np.logical_and(mindepth <= depthdata["depth"], maxdepth >= depthdata["depth"])), "depth"] = None
        grid = depthdata["depth"].reshape(height, width) 

        plt.imshow(grid, extent=(0, width, 0, height)) 
        plt.savefig(args.outframefile + str(framenum).zfill(6) + ".png")
        plt.close(fig)


header = ["frame"] + ["mean" + str(x) for x in range(n_components)] + ["variance" + str(x) for x in range(n_components)] + ["weight" + str(x) for x in range(n_components)]


df = pd.DataFrame(results, columns = header)

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
