import numpy as np
import matplotlib.pyplot as pl
import scipy.special as ss

def BetaDistribution(X, a, b):
    C = ss.gamma(a + b) / (ss.gamma (a) * ss.gamma(b))
    y = (X ** (a - 1) * (1 - X) ** (b - 1)) * C
    return y

if __name__ == '__main__':
   X = np.arange(.01, 1.01, .01)
   pl.figure(figsize = (4, 4))

   pl.plot(X, BetaDistribution(X, 5, 3))
   # pl.plot(X, BetaDistribution(X, 5., 1.))
   # pl.plot(X, BetaDistribution(X, 1., 5.))
   # pl.plot(X, BetaDistribution(X, 5., 2.))
   # pl.plot(X, BetaDistribution(X, 2., 5.))

   pl.xlim(.0, 1.)
   pl.ylim(.0, 2.5)
   pl.show()
