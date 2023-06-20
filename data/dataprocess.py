from csvtonpy import csv_to_npy
from stltonpy import stl_to_npy
import numpy as np
import matplotlib.pyplot as plt

stlsavename="stlsave.npy"
xsavename="xsave.npy"
ysavename="ysave.npy"
vsavename="vsave.npy"

stlsave=np.load(stlsavename)
xsave=np.load(xsavename)
ysave=np.load(ysavename)
vsave=np.load(vsavename)

print(f"stlsave: {stlsave.shape}")
print(f"xsave: {xsave.shape}")
print(f"vsave: {vsave.shape}")
print(f"ysave: {ysave.shape}")

n = int(input("num of files:"))

resolution = 300
stl_filename=input("filename(stl):")+".stl"
stl_ = stl_to_npy(stl_filename, resolution=resolution).astype(np.float32)
stlsave = np.append(stlsave, stl_[np.newaxis,np.newaxis,:,:],axis=0)

if xsave.shape[0] == 0:xidx=0
else: xidx = xsave[-1][0][0][0]+1

x_data = [xidx for _ in range(n)]
x_data = np.array(x_data).astype(np.float32)
xsave = np.append(xsave, x_data[:,np.newaxis,np.newaxis,np.newaxis],axis=0)

csv_filenames = []
for i in range(n):
    filename=input(f"filename{i}(csv):")+".csv"
    csv_filenames.append(filename)

v_data = []
for i in range(n):
    v=float(input(f"velocity{i}(float):"))
    v_data.append(v)
v_data = np.array(v_data).astype(np.float32)
vsave= np.append(vsave, v_data[:,np.newaxis,np.newaxis,np.newaxis], axis=0)

for csv_ in csv_filenames:
    data = csv_to_npy(csv_, resolution=resolution).astype(np.float32)
    for z in range(resolution):
        for x in range(resolution):
            if stl_[z][x] != resolution: data[z][x] = 0
    ysave = np.append(ysave,data[np.newaxis,np.newaxis,:,:],axis=0)

print(f"stlsave: {stlsave.shape}")
print(f"xsave: {xsave.shape}")
print(f"vsave: {vsave.shape}")
print(f"ysave: {ysave.shape}")
np.save("stlsave.npy",stlsave)
np.save("xsave.npy",xsave)
np.save("ysave.npy",ysave)
np.save("vsave.npy",vsave)

while 1:
    end = input("enter to quit")
    if end: break

### Plot Test ###
#ysave = np.load("ysave.npy")
#xsave = np.load("xsave.npy")
#plt.pcolor(xsave[-1],cmap='rainbow')
#plt.pcolor(ysave[-1],cmap='rainbow')
#plt.show()
