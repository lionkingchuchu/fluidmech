import numpy as np

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

stl_d = int(input('stl del num:'))
data_d = int(input('data del num:'))

stlsave = stlsave[:-stl_d]
xsave = xsave[:-data_d]
ysave = ysave[:-data_d]
vsave = vsave[:-data_d]

print(f"stlsave: {stlsave.shape}")
print(f"xsave: {xsave.shape}")
print(f"vsave: {vsave.shape}")
print(f"ysave: {ysave.shape}")

while 1:
    end = input("save:1 quit:0")
    if end=='1':
        np.save("stlsave.npy",stlsave)
        np.save("xsave.npy",xsave)
        np.save("ysave.npy",ysave)
        np.save("vsave.npy",vsave)
        break
    if end=='0':
        break
    


