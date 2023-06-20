import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def csv_to_npy(filename, resolution=600):

    data=np.loadtxt(filename,unpack=True,delimiter=',',skiprows=6).astype(np.float32)

    ycut=3

    data=data.T
    data= sorted(data,key=lambda x:x[1])
    data = np.array(data)

    for i in range(data.shape[0]):
        if data[i][1]>ycut:
            data = data[:i,:]
            break
    print(f"y-cut data nums: {data.shape}")

    data= sorted(data,key=lambda x:x[3])
    data = np.array(data)

    for i in range(data.shape[0]):
        if data[i][3]>0:
            data = data[i:,:]
            break
    print(f"v-cut data nums: {data.shape}")

    data_grid=np.zeros((resolution,resolution))
    data_visit=np.zeros((resolution,resolution),dtype='bool')
    data_floor=np.floor(data[:,:3])
    data_floor=np.concatenate((data_floor,data[:,3].reshape(-1,1)),axis=1)

    for x,y,z,v in data_floor:
        x_=int(x)//(600//resolution)
        z_=int(z)//(600//resolution)
        if x_ == resolution: x_-=1
        if z_ == resolution: z_-=1
        if data_grid[z_][x_]<v:
            data_grid[z_][x_]=v
            data_visit[z_][x_]=True

    fill_blank(data_grid,resolution)

    return data_grid

def fill_blank(data, dim):
    
    que = deque([])
    for z in range(dim):
        for x in range(dim):
            if data[z][x] != 0: que.append((z,x,data[z][x]))
    dz = [-1,1,0,0]
    dx = [0,0,-1,1]

    while que:
        z_,x_,v = que.popleft()
        for i in range(4):
            if 0<=z_+dz[i]<dim and 0<=x_+dx[i]<dim and data[z_+dz[i]][x_+dx[i]] == 0:
                que.append((z_+dz[i],x_+dx[i],v))
                data[z_+dz[i]][x_+dx[i]]=v
    return data.astype(np.float32)

##PLOT TEST##

#filename = input('filename')+'.csv'
#resolution=300
#data_grid= csv_to_npy(filename,resolution=resolution)

#fig = plt.figure(figsize = (9,9))
#ax = plt.axes()
#ax.grid(color='grey',linestyle='-',linewidth=0.3,alpha=0.6)
#sctt = ax.pcolor(data_grid,cmap='rainbow')
#ax.set_xlabel('X-axis', fontweight ='bold')
#ax.set_ylabel('z-axis', fontweight ='bold')
#ax.axis('equal')
#plt.xlim((0,resolution))
#plt.ylim((0,resolution))
#plt.show()
