import numpy as np
import torch
from scipy.spatial.transform import Rotation as RR
import cv2


def center_crop(img, dst_shape):
    src_nr = img.shape[-2]
    src_nc = img.shape[-1]

    dst_nr = dst_shape[-2]
    dst_nc = dst_shape[-1]
    
    if (dst_nr != src_nr) or (dst_nc != src_nc):
        src_start_r = int((src_nr - dst_nr) / 2)
        src_end_r   = src_start_r + dst_nr
        
        src_start_c = int((src_nc - dst_nc) / 2)
        src_end_c   = src_start_c + dst_nc
        
        if img.dim() == 4:
            return img[:,:,src_start_r:src_end_r,src_start_c:src_end_c]
        elif img.dim() == 3:
            return img[:,src_start_r:src_end_r,src_start_c:src_end_c]
        else:
            assert(img.dim() == 2)
            return img[src_start_r:src_end_r,src_start_c:src_end_c]
    else:
        return img

def get_gaussian_2d_heatmap(num_rows, num_cols, sigma, peak_row=None, peak_col=None):
    if peak_row is None:
        peak_row = num_rows // 2
    if peak_col is None:
        peak_col = num_cols // 2
    
    (Y,X) = torch.meshgrid(
        torch.arange(0,num_rows), torch.arange(0,num_cols),
        indexing='ij')
    
    Y = Y.float()
    X = X.float()

    return torch.exp(((X - peak_col).pow(2) + (Y - peak_row).pow(2)) / (sigma * sigma * -2)) / (2 * np.pi * sigma * sigma)


def heatmap2location(heatmap, th=0.80):
    locs = []
    num_landmarks = heatmap.shape[-4]
    landmark_local_template = get_gaussian_2d_heatmap(25, 25, 2.5)
    for idx in range(num_landmarks):
        hm = heatmap[idx,0, ...].detach()
        cur_heat_pad = torch.from_numpy(np.pad(hm.cpu().numpy(), ((12, 12), (12, 12)), 'reflect'))
        max_ind = np.unravel_index(torch.argmax(hm.T).item(), hm.shape[-2:])
        start_roi_row = max_ind[1]
        start_roi_col = max_ind[0]
        heat_roi = cur_heat_pad[start_roi_row:(start_roi_row+25), start_roi_col:(start_roi_col+25)]
        ncc = ncc_2d(landmark_local_template, heat_roi)

        if ncc < th:
            max_ind = (np.inf, np.inf)
        locs.append(max_ind)
    return np.array(locs).astype(np.float32)

def ncc_2d(X,Y):
    N = X.shape[-1] * X.shape[-2]
    assert(N > 1)
    
    #print('X: {}'.format(X.shape))
    #print('Y: {}'.format(Y.shape))

    dim = X.dim()
    d1 = dim - 2
    d2 = dim - 1

    # compute means of each 2D "image"
    mu_X = torch.mean(X, dim=[d1,d2])

    # make the 2D images have zero mean
    X_zm = X - (mu_X.reshape(*mu_X.shape,1,1) * torch.ones_like(X))

    # compute sample standard deviations
    X_sd = torch.sqrt(torch.sum(X_zm * X_zm, dim=[d1,d2]) / (N-1))

    mu_Y = torch.mean(Y, dim=[d1,d2])

    Y_zm = Y - (mu_Y.reshape(*mu_Y.shape,1,1) * torch.ones_like(Y))
    
    Y_sd = torch.sqrt(torch.sum(Y_zm * Y_zm, dim=[d1,d2]) / (N-1))
    
    ncc_losses = torch.sum(X_zm * Y_zm, dim=[d1,d2]) / ((N * (X_sd * Y_sd)) + 1.0e-8)
    return ncc_losses

def extract_scenes(corr, norm=100):
    corr = corr.transpose([1, 2, 0])
    corr_entry = corr[..., :3]*norm
    corr_entry_std = corr[..., 3]
    corr_exit = corr[..., 4:7]*norm
    corr_exit_std = corr[..., 7]
    return corr_entry, corr_entry_std, corr_exit, corr_exit_std

def pnp_ransac(corr, intrinsic,  norm=100, th=0.1, type="both"):
    corr_entry, corr_entry_std, corr_exit, corr_exit_std = extract_scenes(corr, norm=norm)
    u, v = np.meshgrid(np.arange(0, corr.shape[1])+0.5, np.arange(0, corr.shape[2])+0.5)
    image_points = np.stack([u, v], axis=-1)
    if type == "both":
        image_points = np.concatenate([image_points.reshape(-1, 2)[corr_entry_std.reshape(-1) < th], image_points.reshape(-1, 2)[corr_exit_std.reshape(-1) < th]], axis=0)
    elif type == "entry":
        image_points = image_points.reshape(-1, 2)[corr_entry_std.reshape(-1) < th]
    else:
        image_points = image_points.reshape(-1, 2)[corr_exit_std.reshape(-1) < th]


    if type == "both":
        corrs = np.concatenate([corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)
    elif type == "entry":
        corrs = corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th]
    else:
        corrs = corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]
    try:
        success, vector_rotation, vector_translation, inliers = cv2.solvePnPRansac(corrs, image_points , intrinsic, np.array([[0.0, 0.0, 0.0, 0.0]]),
                    reprojectionError=20, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.99)

        r = RR.from_rotvec(vector_rotation[..., 0])
        matrix_rotation = r.as_matrix()
    except Exception as e:
        # Set dummy results
        vector_translation = np.array([[100, 0, 0]]).T
        matrix_rotation = np.identity(3)
        inliers = np.array([0, 1])

    return {
        "R": matrix_rotation,
        "t": vector_translation,
        "inliers": inliers,
        "image_points": image_points,
        "object_points": corrs
    }

def calc_mTRE(vol_landmarks, pred_extrinsic, gt_extrinsic):
    """Calculates mTRE for given landmarks

    Args:
        vol_landmarks (_type_): [N, 3] ndarray
        pred_extrinsic (_type_): [3, 4] ndarray
        gt_extrinsic (_type_): [3, 4] ndarray
    """
    homo_vol_landmarks = np.concatenate([vol_landmarks, 
                                        np.ones([vol_landmarks.shape[0], 1])], axis=-1).T

    pred_transformed_landmarks = pred_extrinsic@homo_vol_landmarks 
    gt_transformed_landmarks = gt_extrinsic@homo_vol_landmarks
    mTRE = np.linalg.norm(pred_transformed_landmarks - gt_transformed_landmarks, axis=0).mean()
    return mTRE

def quaternion_to_matrix(quaternion):
    w, x, y, z = quaternion
    norm = np.linalg.norm(quaternion)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
