import numpy as np

resolution = 300

arr = np.empty((0,1,resolution,resolution)).astype(np.float32)
varr = np.empty((0,1,1,1)).astype(np.float32)
np.save("stlsave.npy",arr)
np.save("xsave.npy",varr)
np.save("ysave.npy",arr)
np.save("vsave.npy",varr)
