import os
from os.path import join
import math
import json

import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import tifffile
import imageio
import pyvista as pv


class DFLDataset(Dataset):
    def __init__(self, datadir, set, pad=True, heatmap=False, hw=[512, 512], no_corrs=False, select_first=None):
        self.IMGDIR = os.path.join(datadir, "images")
        self.CORRDIR = os.path.join(datadir, "correspondences")
        self.ANNODIR = os.path.join(datadir, "annotations")
        self.datadir = datadir
        with open(os.path.join(datadir, set+".json")) as f:
            self.ids = json.load(f)
        self.pad = pad
        self.hw = hw
        self.no_corrs = no_corrs
        self.select_first = select_first
        self.heatmap = heatmap

    def __len__(self):
        if self.select_first is not None:
            return self.select_first
        else:
            return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        img = imageio.imread(os.path.join(self.IMGDIR, f"{id}.png"))
        if not self.no_corrs:
            corr_entry = tifffile.imread(os.path.join(self.CORRDIR, f"{id}_entry.tiff"))
            corr_exit = tifffile.imread(os.path.join(self.CORRDIR, f"{id}_exit.tiff"))
        anno_file_name = os.path.join(self.ANNODIR, f"{id}.json")
        with open(anno_file_name, "r") as f:
            anno = json.load(f)
        img = torch.from_numpy(img)[None, ...]/255.0
        intrinsic = np.array(anno["intrinsic"]["matrix"])
        if self.pad:
            img = F.pad(img, (anno["pad"], anno["pad"]), padding_mode="reflect") 
            intrinsic[0, 2] += anno["pad"]
            intrinsic[1, 2] += anno["pad"]
        else:
            bd = anno["pad"]
            if not self.no_corrs:
                corr_entry = corr_entry[..., bd:-bd+1, bd:-bd+1, :]
                corr_exit = corr_exit[..., bd:-bd+1, bd:-bd+1, :]

        ori_hw = img.shape[1:]
        resize_scale = np.array(self.hw)/ori_hw
        img = F.resize(img, self.hw) 
        if not self.no_corrs:
            corr_entry = torch.permute(torch.from_numpy(corr_entry), [2, 0, 1])
            corr_exit = torch.permute(torch.from_numpy(corr_exit), [2, 0, 1])
            print(self.hw)
            corr_entry = F.resize(corr_entry, self.hw)
            corr_exit = F.resize(corr_exit, self.hw)

        extrinsic = np.array(anno["extrinsic"])
        intrinsic = resize_scale[0]*intrinsic
        intrinsic[2,2] = 1.0

        vol_landmarks = []
        proj_landmarks = []
        for landmark_name in anno["vol-landmarks"].keys():
            vol_landmark = anno["vol-landmarks"][landmark_name]
            proj_landmark = np.array(anno["proj-landmarks"][landmark_name])
            vol_landmarks.append(vol_landmark)
            # insert inf if out of view
            if np.any(proj_landmark < 0) or np.any(proj_landmark > ori_hw):
                proj_landmark = np.array([math.inf, math.inf])
            proj_landmarks.append(proj_landmark)
        
        vol_landmarks = np.array(vol_landmarks)
        proj_landmarks = np.array(proj_landmarks)

        # rescale the proj landmarks
        proj_landmarks = proj_landmarks*resize_scale[None, ...]

        if not self.no_corrs:
            data = {
                "img": img,
                "pad": np.ceil(anno["pad"]*resize_scale).astype(np.int32) if self.pad else np.array([0, 0], dtype=np.int32),
                "vol-landmarks": vol_landmarks,
                "proj-landmarks": proj_landmarks,
                "corr_entry": corr_entry,
                "corr_exit": corr_exit, 
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "carm": anno["intrinsic"],
                "id": id
            }
        else:
            data = {
                "img": img,
                "pad": np.ceil(anno["pad"]*resize_scale).astype(np.int32) if self.pad else np.array([0, 0], dtype=np.int32),
                "vol-landmarks": vol_landmarks,
                "proj-landmarks": proj_landmarks,
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "carm": anno["intrinsic"],
                "id": id
            }
        if self.heatmap:
            hs = get_heatmap(proj_landmarks.T, self.hw)
            data["heatmaps"] = hs

        return data

    def get_mesh(self):
        mesh = pv.read(os.path.join(self.datadir, "pelvis_in_world_coordinate.stl"))
        return mesh


def get_heatmap(landmarks, hw):
    num_lands = landmarks.shape[-1]

    h = torch.zeros(num_lands, 1, hw[0], hw[1])

    # "FH-l", "FH-r", "GSN-l", "GSN-r", "IOF-l", "IOF-r", "MOF-l", "MOF-r", "SPS-l", "SPS-r", "IPS-l", "IPS-r"
    sigma_lut = torch.full([num_lands], 2.5)

    (Y,X) = torch.meshgrid(torch.arange(0, hw[1]),
                            torch.arange(0, hw[0]),
                            indexing='ij')
    Y = Y.float()
    X = X.float()

    for land_idx in range(num_lands):
        sigma = sigma_lut[land_idx]

        cur_land = landmarks[:,land_idx]

        mu_x = cur_land[0]
        mu_y = cur_land[1]

        if not math.isinf(mu_x) and not math.isinf(mu_y):
            pdf = torch.exp(((X - mu_x).pow(2) + (Y - mu_y).pow(2)) / (sigma * sigma * -2)) / (2 * np.pi * sigma * sigma)
            #pdf /= pdf.sum() # normalize to sum of 1
            h[land_idx,0,:,:] = pdf
    
    return h

class DataModule(pl.LightningDataModule):
    def __init__(self, datadir, Dataset, pad=True, heatmap=False, batch_size=16):
        super().__init__()
        self.datadir = datadir
        self.Dataset = Dataset
        self.batch_size = batch_size
        self.pad = pad
        self.heatmap = heatmap

    def setup(self, stage=None):
        self.train_dataset = self.Dataset(self.datadir, "train", self.pad, self.heatmap) 
        self.val_dataset = self.Dataset(self.datadir, "val", self.pad, self.heatmap) 

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=10,
                          pin_memory=False,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=10,
                          pin_memory=False,
                          drop_last=True)

def get_datamodule(datadir, pad=True, heatmap=False, batch_size=1):
    datamodule = DataModule(datadir, DFLDataset, pad=pad, heatmap=heatmap, batch_size=batch_size)
    return datamodule
