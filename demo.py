'''
SANDRO: A Robust Solver with a Splitting Strategy for Point Cloud Registration

Dynamic Legged Systems Lab - Istituto Italiano di Tecnologia

Developed by Michael Adlerstein
Maintened by Michael Adlerstein, João Carlos Virgolino Soares, Angelo Bratta

'''

import open3d as o3d
import numpy as np
import torch
from SANDRO import registration_SANDRO
import scipy.spatial as spatial
import copy
import time

def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T



def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = spatial.cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds
    
def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


if __name__ == "__main__":

    # Load the Demo pointclouds
    VOXEL_SIZE = 0.02
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pca = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    pcb = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])  
    #voxel downsample 
    pca = pca.voxel_down_sample(VOXEL_SIZE)
    pcb = pcb.voxel_down_sample(VOXEL_SIZE)


    #o3d to numpy 
    pointA = np.array(pca.points)
    pointB = np.array(pcb.points)

    # 
    A_xyz = pcd2xyz(pca) # np array of size 3 by N
    B_xyz = pcd2xyz(pcb) # np array of size 3 by M


    # extract FPFH features
    A_feats = extract_fpfh(pca, VOXEL_SIZE)
    B_feats = extract_fpfh(pcb, VOXEL_SIZE)


    corrs_A, corrs_B = find_correspondences(A_feats, B_feats, mutual_filter=True)
    corr_idx = np.vstack((corrs_A, corrs_B))
    A_corr = A_xyz[:, corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B] # np array of size 3 by num_corrs



    s_tensor = torch.tensor(A_corr.T, dtype=torch.float32)
    t_tensor = torch.tensor(B_corr.T, dtype=torch.float32)
    s_tensor = s_tensor.unsqueeze(0).permute(0, 1, 2)
    t_tensor = t_tensor.unsqueeze(0).permute(0, 1, 2)

    start = time.time()
    T_s = registration_SANDRO(s_tensor, t_tensor,  factor = 5 , factor_min=VOXEL_SIZE, stopTh=1e-12)
    finish = time.time()

    print("optimization soled in" , finish- start)
    o3d.visualization.draw_geometries([copy.deepcopy(pca), pcb ])

    pca = pca.paint_uniform_color([1 , 0 , 0 ])
    pcb = pcb.paint_uniform_color([0 , 0 , 1 ])
    o3d.visualization.draw_geometries([copy.deepcopy(pca).transform(T_s), pcb ])
    

