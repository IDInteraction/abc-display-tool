import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
import glob
import loadDepth

np.random.seed(0)

frames = glob.glob("/media/sf_spot_the_difference/depth/P01/depthDepth*.txt")
frames.sort()

print "Using frame ", frames[0], " as reference"

fitframe = loadDepth.loadDepth(frames[0])

# Depths determined from Shiny app; covers as wide a range as possible
# while capturing participant and table
mindepth = 810
maxdepth = 1710

filterdepth = fitframe[np.logical_and(mindepth <= fitframe["depth"] , maxdepth >= fitframe["depth"]) ]

components = np.arange(1,8)

models = [None for i in range(len(components))]

for c in range(len(components)):
    models[c] = mixture.GaussianMixture(n_components = components[c])
    models[c].fit(filterdepth["depth"].reshape(-1,1))
    
BIC = [m.bic(filterdepth["depth"].reshape(-1,1)) for m in models]

n_components = components[np.argmin(BIC)]

print "Fitting all frames with ", n_components, " components"
results = []
for f in frames[1:10]:
    depthdata = loadDepth.loadDepth(f)
    filterdepth = depthdata[np.logical_and(mindepth <= depthdata["depth"], maxdepth >= depthdata["depth"])]
    model = mixture.GaussianMixture(n_components = n_components)
    model.fit(filterdepth["depth"].reshape(-1,1))

    means = [item for sublist in model.means_.tolist() for item  in sublist]  
    covars = [item[0] for sublist in model.covariances_.tolist() for item  in sublist]  
    weights = model.weights_.tolist()

    results.append([f] + means + covars + weights)
    


header = ["frame"] + ["mean" + str(x) for x in range(n_components)] + ["variance" + str(x) for x in range(n_components)] + ["weight" + str(x) for x in range(n_components)]


df = pd.DataFrame(results, columns = header)

print df






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
