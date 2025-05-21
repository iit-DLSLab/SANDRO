'''
SANDRO: A Robust Solver with a Splitting Strategy for Point Cloud Registration

Dynamic Legged Systems Lab - Istituto Italiano di Tecnologia

Developed by Michael Adlerstein
Maintened by Michael Adlerstein, João Carlos Virgolino Soares, Angelo Bratta

'''

import torch 
import numpy as np 
import open3d as o3d
import copy 
import time


def split_cloud(source, n_splitting):
    assert source.size(0) == 1, "Source should have shape [1, N, 3]"
    # Remove the first dimension since it's a singleton
    source = source.squeeze(0)
    
    total_points = source.size(0)
    base_size = total_points // n_splitting
    remainder = total_points % n_splitting
    
    source_subclouds = []
    start_idx = 0
    
    for i in range(n_splitting):
        end_idx = start_idx + base_size + (1 if i < remainder else 0)
        source_subclouds.append(source[start_idx:end_idx].unsqueeze(0))
        start_idx = end_idx
    return source_subclouds





def rigid_transform_3d( A, B, weights, weight_threshold=1e-5 , device = "cuda"):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t 
    """
    # device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    A = A.to(device)
    B = B.to(device)
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0]).to(device)

    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-10)
    # weights[weights < weight_threshold] = 0  # Set weights below threshold to zero        
    # find mean of point cloud
    centroid_A = torch.sum(A * weights.T, dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-10)
    centroid_B = torch.sum(B * weights.T, dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-10)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    # construct weight covariance matrix 
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm
    # find rotation
    U, S, Vt = torch.svd(H.to(device))
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3, dtype=torch.float32)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    T[3, 3] = 1
    return T , R, t



def german_mcclure_loss(x, alpha):
    kernel = (alpha * x**2) / (alpha + x**2)
    weight = alpha / (alpha + x**2)**2
    return weight, kernel




def registration_SANDRO(source , target , split = 1 , factor = 5 , factor_min=0.05, decay = 0.9 ,  stopTh = 1e-10):
    """
    initialize with the split and remerge at the end only the best idx
    1- split into subcloud and perform registraiton on each subcloud
    2 - use german macclure as loss, with GNC or fixed 
    """
    source_subclouds = split_cloud(source , n_splitting=split)
    target_subclouds = split_cloud(target , n_splitting=split)
    n_subclouds = len(target_subclouds)
    # Create a vector of ones with the same length as source[0]
    vector_ones = torch.ones(source_subclouds[0].shape[1]).unsqueeze(0).to(dtype=torch.float32)
    matrix_weights = []
    # Repeat the vector to create a matrix with n rows
    for subcloud in source_subclouds:
        matrix_weights.append(torch.ones(subcloud[:,:,0].shape))
    # matrix_weights = [torch.ones(subcloud.size(), dtype=torch.float32) for subcloud in source_subclouds[:,:,0]]
    Ts= torch.ones(n_subclouds, 4, 4 )
    Rs = torch.ones(n_subclouds, 3, 3)
    ts = torch.ones(n_subclouds, 3, 1)
    costs_prev = torch.zeros(split , 1 )
    costs = torch.ones(split , 1 )
    residuals_s = [] 
    stop_idx = []
    loss = torch.ones(n_subclouds , 1 , 1 )
    loss_final = torch.ones(n_subclouds , 1 , 1 )

    for i in range(100):
        for n_cloud in range(n_subclouds):
            if n_cloud in stop_idx:
                continue
            T , R  ,t  = rigid_transform_3d(source_subclouds[n_cloud] , target_subclouds[n_cloud] ,  matrix_weights[n_cloud])
            Ts[n_cloud] = T
            Rs[n_cloud] = R 
            ts[n_cloud] = t 
            source_trans = source_subclouds[n_cloud]  @ torch.inverse(Rs[n_cloud]) +  ts[n_cloud].reshape(1 , 3)
            residuals = torch.norm(source_trans - target_subclouds[n_cloud] , dim = 2)
            residuals_s.append(residuals)  
            w , rho = german_mcclure_loss(residuals , factor)
            
            if factor > factor_min:
                factor = factor*(0.90)
            else :
                factor = factor_min

            costs[n_cloud] = (torch.sum(rho))
            matrix_weights[n_cloud] = w

            #this part checks if all the subclouds have converged in a minima,
            #stop the function only if all the subclouds have converged
            if  abs(costs[n_cloud] - costs_prev[n_cloud]) <= abs(stopTh) :
                stop_idx.append(n_cloud)
                if len(stop_idx) == split:
                    best_subcloud = torch.argmin(costs)  
                    print(f"solution after {i} iterations")
                    return Ts[best_subcloud]   
                 
            costs_prev[n_cloud] = costs[n_cloud]
    best_subcloud = torch.argmin(costs)
    print(f"solution after {i} iterations")
    return Ts[best_subcloud]      
