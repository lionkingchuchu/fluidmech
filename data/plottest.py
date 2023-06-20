from csvtonpy import csv_to_npy
from stltonpy import stl_to_npy
import numpy as np
import matplotlib.pyplot as plt

resolution = 300
### Plot Test ###
ysave = np.load("ysave.npy")
xsave = np.load("xsave.npy")
fig = plt.figure(figsize = (10,9))
ax = plt.axes()
sctt = ax.pcolor(ysave[-1][0],cmap='rainbow')
fig.colorbar(sctt)
ax.axis('equal')
plt.xlim((0,resolution))
plt.ylim((0,resolution))
plt.show()

print(xsave[-1])
