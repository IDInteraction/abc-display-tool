import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture

import loadDepth


np.random.seed(0)

fitframe = loadDepth.loadDepth("/media/sf_spot_the_difference/depth/P01/depthDepth000001.txt")

mindepth = 810
maxdepth = 1710


filterdepth = fitframe[np.logical_and(mindepth <= fitframe["depth"] , maxdepth >= fitframe["depth"]) ]

clf = mixture.GaussianMixture(n_components = 5)
clf.fit(filterdepth["depth"].reshape(-1,1))

x = np.linspace(mindepth, maxdepth).reshape(-1,1)

logprob = clf.score_samples(x)
pdf = np.exp(logprob)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(filterdepth["depth"].reshape(-1,1), 30, normed=True, histtype='stepfilled', alpha=0.4)
ax.plot(x, pdf)

print "Means"
print clf.means_
print "Covariance"
print clf.covariances_
print "Weight"
print clf.weights_

plt.show()
