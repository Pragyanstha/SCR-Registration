import os
import time

import numpy as np
import cv2
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation as RR

from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
import imageio

from utils.model_helpers import heatmap2location, pnp_ransac, extract_scenes, calc_mTRE, quaternion_to_matrix
from utils.misc import save_ply


def evaluate_posenet(data, model, OUTDIR, config, mesh=None, idx=0):
    save_overlay = config["save_overlay"]
    img = data["img"]
    gt_extrinsic = data["extrinsic"][:-1, :]
    vol_landmarks = data["vol-landmarks"]
    t1 = time.time()
    with torch.no_grad():
        pred_pos, pred_ori, _ = model(img[None, ...].cuda())
    t2 = time.time()
    elapsed = t2 - t1
    R = quaternion_to_matrix(pred_ori[0].cpu().numpy())
    t =  pred_pos[0][..., None].cpu().numpy()
    pred_extrinsic = np.concatenate([R, t], axis=-1)

    mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
    error_R = 180/np.pi*np.arccos(
        np.clip((np.trace(R.T@gt_extrinsic[:, :-1])- 1)/2.0, -1.0, 1.0)
        )
    error_t = np.linalg.norm(gt_extrinsic[:, -1] - t[:, 0])
    print(f"[{data['id']} - {idx+1}] Rot err: {error_R}, Translation err : {error_t}, mTRE: {mTRE}")
    res = {
        "pred_extrinsic": pred_extrinsic,
        "gt_extrinsic": gt_extrinsic,
        "rotation_error": error_R,
        "translation_error": error_t,
        "mTRE": mTRE,
        "time": elapsed
    }

    if mesh is not None and save_overlay:
        print("Viz in progress")
        s_w = data["carm"]["sensor-width"] * data["carm"]["pixel-size"]
        s_h = data["carm"]["sensor-height"] * data["carm"]["pixel-size"]
        dsd = data["carm"]["source-to-detector-distance"]
        detector_plane = pv.Plane(
                center=[0, 0, dsd],
                direction=[0, 0, -1],
                i_size=s_w,
                j_size=s_h)
        detector_plane.texture_map_to_plane(
                    origin=[-s_w/2.0, s_h/2.0, dsd],
                    point_u=[s_w/2.0, s_h/2.0, dsd],
                    point_v=[-s_w/2.0, -s_h/2.0, dsd], inplace=True)
        nodes = [
            [0, 0, 0],
            [s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, -s_h/2.0, dsd],
            [s_w/2.0, -s_h/2.0, dsd],
        ]
        edges = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
        ])
        # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(edges.shape[0], int) * 2
        padding[:] = 2
        edges_w_padding = np.vstack((padding, edges.T)).T
        img_padd = data["pad"]
        wires = pv.PolyData(nodes, edges_w_padding)
        colors = range(edges.shape[0])
        if img_padd[0] != 0:
            tex = gray2rgb(img.numpy()[0, img_padd[0]:-img_padd[0]+1, img_padd[1]:-img_padd[1]+1])*255
        else:
            tex = gray2rgb(img.numpy()[0])*255
        tex = pv.numpy_to_texture(rescale_intensity(tex, out_range=(0, 255)).astype(np.uint8))
        camera = [(300, -150, -600), (0, 0, dsd/2.0), (0, -1, 0)]
        gt_transform = np.concatenate([gt_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        pred_transform = np.concatenate([pred_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2048, 1024])
        plotter.set_background("white")
        plotter.subplot(0,0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         color="green",
                         opacity=0.5)
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         color="red",
                         opacity=0.8)
        plotter.add_mesh(wires, 
                         scalars=colors,
                         render_lines_as_tubes=True,
                         style="wireframe",
                         line_width=3,
                         show_scalar_bar=False,
                         cmap="jet")
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.subplot(0, 1)
        camera = [(0, 0, 0), (0, 0, dsd), (0, -1, 0)]
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         silhouette={
                            "color": "red",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         silhouette={
                            "color": "green",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.camera.view_angle = np.rad2deg(2*np.arctan(s_h/(2.0*dsd)))
        plotter.show(screenshot=os.path.join(OUTDIR, "mesh.png"))

    return res

def evaluate_scrnet(data, model, OUTDIR, config, mesh=None, idx=0):
    th = config["th"]
    save_pointcloud = config["save_pointcloud"]
    save_overlay = config["save_overlay"]
    type = config["type"]
    img = data["img"]
    gt_extrinsic = data["extrinsic"][:-1, :]
    gt_intrinsic = data["intrinsic"]
    vol_landmarks = data["vol-landmarks"]
    gt_corr_entry = data["corr_entry"].numpy().transpose([1, 2, 0])
    gt_corr_exit = data["corr_exit"].numpy().transpose([1, 2, 0])

    t1 = time.time()
    corr = model(img[None, ...].cuda())[0].detach().cpu().numpy()

    corr_entry, corr_entry_std, corr_exit, corr_exit_std = extract_scenes(corr)

    if save_pointcloud:
        save_ply(corr_entry.reshape(-1, 3), os.path.join(OUTDIR, "pred_corr_entry.ply"), np.tile(corr_entry_std.reshape(-1), (3, 1)).T)
        save_ply(gt_corr_entry.reshape(-1, 3), os.path.join(OUTDIR, "gt_corr_entry.ply"))

        save_ply(corr_exit.reshape(-1, 3), os.path.join(OUTDIR, "pred_corr_exit.ply"), np.tile(corr_exit_std.reshape(-1), (3, 1)).T)
        save_ply(gt_corr_exit.reshape(-1, 3), os.path.join(OUTDIR, "gt_corr_exit.ply"))

    pose_dict = pnp_ransac(corr, gt_intrinsic, model.norm, th=th, type=type)
    t2 = time.time()
    elapsed = t2 - t1
    R = pose_dict["R"]
    t = pose_dict["t"]
    pred_extrinsic = np.concatenate([R, t], axis=-1)

    mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
    error_R = 180/np.pi*np.arccos(
        np.clip((np.trace(R.T@gt_extrinsic[:, :-1])- 1)/2.0, -1.0, 1.0)
        )
    error_t = np.linalg.norm(gt_extrinsic[:, -1] - t[:, 0])
    print(f"[{data['id']} - {idx+1}] Rot err: {error_R}, Translation err : {error_t}, mTRE: {mTRE}")
    res = {
        "pred_extrinsic": pred_extrinsic,
        "gt_extrinsic": gt_extrinsic,
        "rotation_error": error_R,
        "translation_error": error_t,
        "mTRE": mTRE,
        "time": elapsed
    }

    if mesh is not None and save_overlay:
        print("Viz in progress")
        s_w = data["carm"]["sensor-width"] * data["carm"]["pixel-size"]
        s_h = data["carm"]["sensor-height"] * data["carm"]["pixel-size"]
        dsd = data["carm"]["source-to-detector-distance"]
        detector_plane = pv.Plane(
                center=[0, 0, dsd],
                direction=[0, 0, -1],
                i_size=s_w,
                j_size=s_h)
        detector_plane.texture_map_to_plane(
                    origin=[-s_w/2.0, s_h/2.0, dsd],
                    point_u=[s_w/2.0, s_h/2.0, dsd],
                    point_v=[-s_w/2.0, -s_h/2.0, dsd], inplace=True)
        nodes = [
            [0, 0, 0],
            [s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, -s_h/2.0, dsd],
            [s_w/2.0, -s_h/2.0, dsd],
        ]
        edges = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
        ])
        # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(edges.shape[0], int) * 2
        padding[:] = 2
        edges_w_padding = np.vstack((padding, edges.T)).T
        img_padd = data["pad"]
        wires = pv.PolyData(nodes, edges_w_padding)
        colors = range(edges.shape[0])
        if img_padd[0] != 0:
            tex = gray2rgb(img.numpy()[0, img_padd[0]:-img_padd[0]+1, img_padd[1]:-img_padd[1]+1])*255
        else:
            tex = gray2rgb(img.numpy()[0])*255
        tex = pv.numpy_to_texture(rescale_intensity(tex, out_range=(0, 255)).astype(np.uint8))
        camera = [(300, -150, -600), (0, 0, dsd/2.0), (0, -1, 0)]
        gt_transform = np.concatenate([gt_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        pred_transform = np.concatenate([pred_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2048, 1024])
        plotter.set_background("white")
        plotter.subplot(0,0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         color="green",
                         opacity=0.5)
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         color="red",
                         opacity=0.8)
        plotter.add_mesh(wires, 
                         scalars=colors,
                         render_lines_as_tubes=True,
                         style="wireframe",
                         line_width=3,
                         show_scalar_bar=False,
                         cmap="jet")
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.subplot(0, 1)
        camera = [(0, 0, 0), (0, 0, dsd), (0, -1, 0)]
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         silhouette={
                            "color": "red",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         silhouette={
                            "color": "green",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.camera.view_angle = np.rad2deg(2*np.arctan(s_h/(2.0*dsd)))
        plotter.show(screenshot=os.path.join(OUTDIR, "mesh.png"))

    return res

def evaluate_dflnet(data, model, OUTDIR, config, mesh=None, idx=0):
    save_heatmaps = config["save_heatmaps"]
    save_overlay = config["save_overlay"]
    img = data["img"].cuda()
    gt_extrinsic = data["extrinsic"][:-1, :]
    gt_intrinsic = data["intrinsic"]
    gt_heatmap = data["heatmaps"].cpu().numpy()
    gt_proj_landmarks = data["proj-landmarks"]
    vol_landmarks = data["vol-landmarks"]

    t1 = time.time()
    pred_heatmap = model(img[None, ...])[0].unsqueeze(-3)
    pred_proj_landmarks = heatmap2location(pred_heatmap)
    pred_heatmap = pred_heatmap.detach().cpu().numpy()

    if save_heatmaps:
        overlay_img = gray2rgb(rescale_intensity(img[0].numpy(), out_range=(0, 255)).astype(np.uint8))
        overlay_img_gt = overlay_img.copy()
        overlay_img_pred = overlay_img.copy()
        overlay_img_pred[..., 0] += rescale_intensity(255-pred_heatmap.sum(axis=0)[0], out_range=(0, 255)).astype(np.uint8)
        overlay_img_gt[..., 0] += rescale_intensity(gt_heatmap.sum(axis=0)[0], out_range=(0, 255)).astype(np.uint8)
        os.makedirs(os.path.join(OUTDIR, "heatmaps"), exist_ok=True)
        for h_idx in range(14):
            imageio.imwrite(os.path.join(OUTDIR, "heatmaps", f"phm_{h_idx}.png"), pred_heatmap[h_idx][0]*255)
            imageio.imwrite(os.path.join(OUTDIR, "heatmaps", f"ghm_{h_idx}.png"), gt_heatmap[h_idx][0]*255)
    
        imageio.imwrite(os.path.join(OUTDIR, "pred_heatmaps.png"), overlay_img_pred) 
        imageio.imwrite(os.path.join(OUTDIR, "gt_hetamaps.png"), overlay_img_gt)
        np.savetxt(os.path.join(OUTDIR, "gt_landmarks.csv"), gt_proj_landmarks, delimiter=",")
        np.savetxt(os.path.join(OUTDIR, "pred_landmarks.csv"), pred_proj_landmarks, delimiter=",")
    
    selections =  ~np.any(np.isinf(pred_proj_landmarks), axis=-1)
    try:
        success, vector_rotation, vector_translation, inliers = cv2.solvePnPRansac(
            vol_landmarks[selections], pred_proj_landmarks[selections] , gt_intrinsic, np.array([[0.0, 0.0, 0.0, 0.0]]),
            reprojectionError=20, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
        t2 = time.time()
        elapsed = t2 - t1
    except Exception as e:
        print(f"[{data['id']} - {idx+1}] registration failed; Num points - {np.count_nonzero(selections)}  ")
        return {
        "pred_extrinsic": np.nan,
        "gt_extrinsic": gt_extrinsic,
        "rotation_error": np.nan,
        "translation_error": np.nan,
        "mTRE": np.nan
        }
    r = RR.from_rotvec(vector_rotation[..., 0])
    pred_extrinsic = np.concatenate([r.as_matrix(), vector_translation], axis=-1)

    pred_extrinsic = np.concatenate([r.as_matrix(), vector_translation], axis=-1)

    mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
    
    error_R = 180/np.pi*np.arccos(
        np.clip((np.trace(r.as_matrix().T@gt_extrinsic[:, :-1])- 1)/2.0, -1.0, 1.0)
        )
    error_t = np.linalg.norm(gt_extrinsic[:, -1] - vector_translation[:, 0])
    print(f"[{data['id']} - {idx+1}] Rot err: {error_R}, Translation err : {error_t}, mTRE: {mTRE}")
    res = {
        "pred_extrinsic": pred_extrinsic,
        "gt_extrinsic": gt_extrinsic,
        "rotation_error": error_R,
        "translation_error": error_t,
        "mTRE": mTRE,
        "time": elapsed
    }

    if mesh is not None and save_overlay:
        print("Viz in progress")
        s_w = data["carm"]["sensor-width"] * data["carm"]["pixel-size"]
        s_h = data["carm"]["sensor-height"] * data["carm"]["pixel-size"]
        dsd = data["carm"]["source-to-detector-distance"]
        detector_plane = pv.Plane(
                center=[0, 0, dsd],
                direction=[0, 0, -1],
                i_size=s_w,
                j_size=s_h)
        detector_plane.texture_map_to_plane(
                    origin=[-s_w/2.0, s_h/2.0, dsd],
                    point_u=[s_w/2.0, s_h/2.0, dsd],
                    point_v=[-s_w/2.0, -s_h/2.0, dsd], inplace=True)
        nodes = [
            [0, 0, 0],
            [s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, s_h/2.0, dsd],
            [-s_w/2.0, -s_h/2.0, dsd],
            [s_w/2.0, -s_h/2.0, dsd],
        ]
        edges = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
        ])
        # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(edges.shape[0], int) * 2
        padding[:] = 2
        edges_w_padding = np.vstack((padding, edges.T)).T
        img_padd = data["pad"]
        wires = pv.PolyData(nodes, edges_w_padding)
        colors = range(edges.shape[0])
        if img_padd[0] != 0:
            tex = gray2rgb(img.numpy()[0, img_padd[0]:-img_padd[0]+1, img_padd[1]:-img_padd[1]+1])*255
        else:
            tex = gray2rgb(img.numpy()[0])*255
        tex = pv.numpy_to_texture(rescale_intensity(tex, out_range=(0, 255)).astype(np.uint8))
        camera = [(300, -150, -600), (0, 0, dsd/2.0), (0, -1, 0)]
        gt_transform = np.concatenate([gt_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        pred_transform = np.concatenate([pred_extrinsic, np.array([[0, 0, 0, 1.0]])], axis=0)
        
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2048, 1024])
        plotter.set_background("white")
        plotter.subplot(0,0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         color="green",
                         opacity=0.5)
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         color="red",
                         opacity=0.8)
        plotter.add_mesh(wires, 
                         scalars=colors,
                         render_lines_as_tubes=True,
                         style="wireframe",
                         line_width=3,
                         show_scalar_bar=False,
                         cmap="jet")
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.subplot(0, 1)
        camera = [(0, 0, 0), (0, 0, dsd), (0, -1, 0)]
        plotter.add_mesh(mesh.transform(gt_transform, inplace=False),
                         silhouette={
                            "color": "red",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(mesh.transform(pred_transform, inplace=False),
                         silhouette={
                            "color": "green",
                            "line_width": 6,
                            "opacity": 0.3
                         },
                         opacity=0)
        plotter.add_mesh(detector_plane, texture=tex)
        plotter.camera.position = camera[0]
        plotter.camera.focal_point = camera[1]
        plotter.camera.up = camera[2]
        plotter.camera.view_angle = np.rad2deg(2*np.arctan(s_h/(2.0*dsd)))
        plotter.show(screenshot=os.path.join(OUTDIR, "mesh.png"))

    return res
