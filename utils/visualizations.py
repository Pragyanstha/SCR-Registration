import os

import numpy as np
import pyvista as pv

from utils.model_helpers import extract_scenes

def get_visualizer(type):
    return viz_types[type] 


def coordmap(model, dataset, config):
    idx = config["idx"]
    th = config["th"]
    mesh = dataset.get_mesh()
    data = dataset[idx]
    img = data["img"]
    vol_landmarks = data["vol-landmarks"]
    gt_intrinsic = data["intrinsic"]
    gt_extrinsic = data["extrinsic"][:-1]
    gt_corr_entry = data["corr_entry"].numpy().transpose([1,2,0])
    gt_corr_exit = data["corr_exit"].numpy().transpose([1,2,0])
    corrs = model(img[None,...]).cpu().detach().numpy()[0]
    corr_entry, corr_entry_std, corr_exit, corr_exit_std = extract_scenes(corrs)

    corrs = np.concatenate([corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True)

    r = 500
    off_x = -100
    off_z = -100
    camera = [(off_x, 0, r + off_z), (off_x, 0, off_z), (-1, 0, 0)]
    points = corrs.reshape(-1, 3)
    rgba = points
    rgba /= rgba.max(axis=0)
    plotter.add_points(points, scalars=rgba, rgba=True, render_points_as_spheres=True, point_size=5)

    rgba = points
    rgba /= rgba.max(axis=0)

    plotter.camera.position = camera[0]
    plotter.camera.focal_point = camera[1]
    plotter.camera.up = camera[2]
    plotter.add_axes(line_width=5)

    plotter.open_gif(os.path.join(config["outdir"], f"COORDMAP_{config['vol']}_{idx}.gif"))

    nframe = 180
    print("Writing to gif...")
    for theta in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
        plotter.camera.position = (off_x, r*np.sin(theta), r*np.cos(theta)+off_z)
        # Write a frame. This triggers a render.
        plotter.write_frame()
    print("Completed")

def errormap(model, dataset, config):
    idx = config["idx"]
    th = config["th"]
    mesh = dataset.get_mesh()
    data = dataset[idx]
    img = data["img"]
    vol_landmarks = data["vol-landmarks"]
    gt_intrinsic = data["intrinsic"]
    gt_extrinsic = data["extrinsic"][:-1]
    gt_corr_entry = data["corr_entry"].numpy().transpose([1,2,0])
    gt_corr_exit = data["corr_exit"].numpy().transpose([1,2,0])
    corrs = model(img[None,...]).cpu().detach().numpy()[0]
    corr_entry, corr_entry_std, corr_exit, corr_exit_std = extract_scenes(corrs)

    corrs = np.concatenate([corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)
    gt_objectpoints = np.concatenate([gt_corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], gt_corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)
    error_map = np.linalg.norm(corrs - gt_objectpoints, axis=-1).reshape(-1)
    # error_map = rescale_intensity(error_map, out_range=(0, 1))
    error_map = np.clip(error_map, None, 10)
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True)

    r = 500
    off_x = -100
    off_z = -100
    camera = [(off_x, 0, r + off_z), (off_x, 0, off_z), (-1, 0, 0)]
    points = corrs.reshape(-1, 3)

    plotter.add_points(points, scalars=error_map,
                       cmap="magma", render_points_as_spheres=True,
                       show_scalar_bar = False,
                    point_size=5)

    plotter.camera.position = camera[0]
    plotter.camera.focal_point = camera[1]
    plotter.camera.up = camera[2]
    plotter.add_axes(line_width=5)
    plotter.add_scalar_bar("L2 error [mm] (Clipped to 10mm)")

    plotter.open_gif(os.path.join(config["outdir"], f"ERRORMAP_{config['vol']}_{idx}.gif"))

    nframe = 180
    print("Writing to gif...")
    for theta in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
        plotter.camera.position = (off_x, r*np.sin(theta), r*np.cos(theta)+off_z)
        # Write a frame. This triggers a render.
        plotter.write_frame()
    print("Completed")

def stdmap(model, dataset, config):
    idx = config["idx"]
    th = config["th"]
    mesh = dataset.get_mesh()
    data = dataset[idx]
    img = data["img"]
    vol_landmarks = data["vol-landmarks"]
    gt_intrinsic = data["intrinsic"]
    gt_extrinsic = data["extrinsic"][:-1]
    gt_corr_entry = data["corr_entry"].numpy().transpose([1,2,0])
    gt_corr_exit = data["corr_exit"].numpy().transpose([1,2,0])
    corrs = model(img[None,...]).cpu().detach().numpy()[0]
    corr_entry, corr_entry_std, corr_exit, corr_exit_std = extract_scenes(corrs)

    corrs = np.concatenate([corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)

    std_map = np.concatenate([corr_entry_std.reshape(-1)[corr_entry_std.reshape(-1) < th],
                              corr_exit_std.reshape(-1)[corr_exit_std.reshape(-1) < th]], axis=0)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True)

    r = 500
    off_x = -100
    off_z = -100
    camera = [(off_x, 0, r + off_z), (off_x, 0, off_z), (-1, 0, 0)]
    points = corrs.reshape(-1, 3)

    plotter.add_points(points, scalars=std_map,
                       cmap="magma", render_points_as_spheres=True,
                       show_scalar_bar = False,
                    point_size=5)

    plotter.camera.position = camera[0]
    plotter.camera.focal_point = camera[1]
    plotter.camera.up = camera[2]
    plotter.add_axes(line_width=5)
    plotter.add_scalar_bar("Log variance")

    plotter.open_gif(os.path.join(config["outdir"], f"STDMAP_{config['vol']}_{idx}.gif"))

    nframe = 180
    print("Writing to gif...")
    for theta in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
        plotter.camera.position = (off_x, r*np.sin(theta), r*np.cos(theta)+off_z)
        # Write a frame. This triggers a render.
        plotter.write_frame()
    print("Completed")
   

viz_types = {
    "coordmap": coordmap,
    "errormap": errormap,
    "stdmap": stdmap
}
