import stltovoxel
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt

def convert_mesh(mesh, resolution=100, voxel_size=None, parallel=True):
    return convert_meshes([mesh], resolution, voxel_size, parallel)


def convert_meshes(meshes, resolution=100, voxel_size=None, parallel=True):
    scale, shift, shape = stltovoxel.slice.calculate_scale_shift(meshes, resolution, voxel_size)
    vol = np.zeros(shape[::-1], dtype=np.int8)

    for mesh_ind, org_mesh in enumerate(meshes):
        stltovoxel.slice.scale_and_shift_mesh(org_mesh, scale, shift)
        cur_vol = stltovoxel.slice.mesh_to_plane(org_mesh, shape, parallel)
        vol[cur_vol] = mesh_ind + 1
    return vol, scale, shift


def convert_file(input_file_path, output_file_path, resolution=100, voxel_size=None, pad=1, parallel=False):
    return convert_files([input_file_path], output_file_path, resolution=resolution,
                  voxel_size=voxel_size, pad=pad, parallel=parallel)


def convert_files(input_file_paths, output_file_path, colors=[(255, 255, 255)],
                  resolution=100, voxel_size=None, pad=1, parallel=False):
    meshes = []
    for input_file_path in input_file_paths:
        mesh_obj = mesh.Mesh.from_file(input_file_path)
        org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
        meshes.append(org_mesh)

    vol, scale, shift = convert_meshes(meshes, resolution, voxel_size, parallel)
    return vol


def stl_to_npy(filename, resolution=600):
    vol = convert_file(filename,'__',resolution=resolution,pad=0)
    vol_2d=np.sum(vol,axis=1)
    vol_2d=vol_2d[:-1,:-1]
    print("stl complete")
    return vol_2d.astype(np.float32)
