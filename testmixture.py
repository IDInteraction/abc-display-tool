import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

n_samples=3000
np.random.seed(0)

gaussian1 = 0.5 * np.random.randn(n_samples, 1) + 2
gaussian2 = np.random.randn(n_samples , 1) - 2

fitdata = np.vstack([gaussian1, gaussian2])

clf = mixture.GaussianMixture(n_components = 2)
clf.fit(fitdata)

x = np.linspace(-4,4)

y = -clf.score_samples(x.reshape(-1,1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(fitdata, 30, normed=True, histtype='stepfilled', alpha=0.4)
ax.plot(x, -y)

#plt.show()
print "Means"
print clf.means_
print "Covariance"
print clf.covariances_
print "Weight"
print clf.weights_

