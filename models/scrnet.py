""" Parts of the U-Net model """
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation as RR

from models.unet import UNet
from utils.model_helpers import pnp_ransac, calc_mTRE


class LitSCRNet(pl.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.unet = UNet(1, 8)
        self.norm = model_cfg["norm"]
        self.lr = model_cfg["lr"]
        self.save_hyperparameters()
        self.transforms = transforms.Compose([
            transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=1, contrast=1),
                transforms.RandomInvert()
            ]
            ),
            transforms.RandomErasing()
        ]
        )

    def calc_loss(self, preds, corrs_entry, corrs_exit):
        preds = preds.permute([0, 2, 3, 1])
        corrs_entry = corrs_entry.permute([0, 2, 3, 1])
        corrs_exit = corrs_exit.permute([0, 2, 3, 1])
        B = preds.shape[0]
        pred_mean_entry = preds[..., :, :, :3].reshape(B, -1, 3)
        log_pred_std_entry = preds[..., :, :, 3].reshape(B, -1, 1)
        pred_std_entry = torch.exp(log_pred_std_entry)
        pred_mean_exit = preds[..., :, :, 4:7].reshape(B, -1, 3)
        log_pred_std_exit = preds[..., :, :, 7].reshape(B, -1, 1)
        pred_std_exit = torch.exp(log_pred_std_exit)
        corrs_entry = corrs_entry.reshape(B, -1, 3)
        corrs_exit = corrs_exit.reshape(B, -1, 3)
        corrs_entry_mask = torch.all(corrs_entry != 0, dim=-1)
        corrs_exit_mask = torch.all(corrs_exit != 0, dim=-1)

        loss_entry_positive = ((pred_mean_entry[corrs_entry_mask] - corrs_entry[corrs_entry_mask])**2).sum(dim=-1, keepdims=True)/(pred_std_entry[corrs_entry_mask]**2)
        loss_entry_positive = (loss_entry_positive + 2*log_pred_std_entry[corrs_entry_mask]).mean()
        
        loss_exit_positive = ((pred_mean_exit[corrs_exit_mask] - corrs_exit[corrs_exit_mask])**2).sum(dim=-1, keepdims=True)/(pred_std_exit[corrs_exit_mask]**2)
        loss_exit_positive = (loss_exit_positive + 2*log_pred_std_exit[corrs_exit_mask]).mean()
        corrs_entry_mask_not = ~corrs_entry_mask
        corrs_exit_mask_not = ~corrs_exit_mask
        loss_entry_negative = 0
        loss_exit_negative = 0
        if torch.any(corrs_entry_mask_not): 
            loss_entry_negative = (torch.exp(-pred_std_entry[corrs_entry_mask_not])).mean()
        if torch.any(corrs_exit_mask_not): 
            loss_exit_negative = (torch.exp(-pred_std_exit[~corrs_exit_mask])**2).mean()

        loss = loss_entry_positive + loss_entry_negative + loss_exit_positive + loss_exit_negative
        return {
            "loss": loss,
            "loss_entry_positive": loss_entry_positive,
            "loss_entry_negative": loss_entry_negative,
            "loss_exit_positive": loss_exit_positive,
            "loss_exit_negative": loss_exit_negative
            }

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        if self.transforms is not None:
            imgs = self.transforms(imgs)

        corrs_entry = batch["corr_entry"]/self.norm
        corrs_exit = batch["corr_exit"]/self.norm
        pred_corrs = self.forward(imgs)
        loss_dict  = self.calc_loss(pred_corrs, corrs_entry, corrs_exit)
        self.log_dict(loss_dict)
        return {"loss": loss_dict["loss"], "predictions": pred_corrs}
    
    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        corrs_entry = batch["corr_entry"]/self.norm
        corrs_exit = batch["corr_exit"]/self.norm
        pad = batch["pad"]
        pred_corrs = self.forward(imgs)
        loss_dict = self.calc_loss(pred_corrs, corrs_entry, corrs_exit)

        # Calculatet mTRE for the first batch only
        if batch_idx == 0:
            for sample_idx in range(imgs.shape[0]):
                intrinsic = batch["intrinsic"][sample_idx].cpu().numpy()
                gt_extrinsic = batch["extrinsic"][sample_idx][:-1, :].cpu().numpy()
                vol_landmarks = batch["vol-landmarks"][sample_idx].cpu().numpy()
                corr = pred_corrs[sample_idx].cpu().numpy()
                pose_dict = pnp_ransac(corr, intrinsic=intrinsic, norm=self.norm, th=0)
                pred_extrinsic = np.concatenate([pose_dict["R"], pose_dict["t"]], axis=-1)
                mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
                self.log("val/mTRE", mTRE)
            return {
                "loss": loss_dict["loss"],
                "imgs": imgs,
                "predictions": pred_corrs[:, :3], 
                "groundtruth": corrs_entry, 
                "pad": pad[0]
                }    
            
        self.log("val/loss", loss_dict["loss"])
        return { 
                "loss": loss_dict["loss"]
        }
    
    def validation_epoch_end(self, outputs):
        pred = outputs[0]["predictions"]
        gt = outputs[0]["groundtruth"]
        img = outputs[0]["imgs"]
        pad = outputs[0]["pad"]
        if pad[0] != 0:
            print("padding detected")
            roi_img = torch.zeros_like(img) 
            roi_img[..., pad[0]:-pad[0]+1, pad[1]:-pad[1]+1] = img[..., pad[0]:-pad[0]+1, pad[1]:-pad[1]+1]
        else:
            roi_img = img
        self.logger.log_image(key="samples", images=[roi_img, pred, gt])

    def forward(self, inputs):
        return self.unet(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                "monitor": "val/loss"
                }

    def predict_pose(self, img, gt_intrinsic, th=0):
        img = img[None, ...]
        corr = self(img)
        corr_entry = torch.permute(corr.squeeze(), [1, 2, 0]).detach().cpu().numpy()[..., :3]*self.norm
        corr_entry_std = torch.permute(corr.squeeze(), [1, 2, 0]).detach().cpu().numpy()[..., 3]
        corr_exit = torch.permute(corr.squeeze(), [1, 2, 0]).detach().cpu().numpy()[..., 4:7]*self.norm
        corr_exit_std = torch.permute(corr.squeeze(), [1, 2, 0]).detach().cpu().numpy()[..., 7]
        u, v = np.meshgrid(np.arange(0, img.shape[2])+0.5, np.arange(0, img.shape[3])+0.5)
        image_points = np.stack([u, v], axis=-1)
        image_points = np.concatenate([image_points.reshape(-1, 2)[corr_entry_std.reshape(-1) < th], image_points.reshape(-1, 2)[corr_exit_std.reshape(-1) < th]], axis=0)
        corrs = np.concatenate([corr_entry.reshape(-1, 3)[corr_entry_std.reshape(-1) < th], corr_exit.reshape(-1, 3)[corr_exit_std.reshape(-1) < th]], axis=0)
        success, vector_rotation, vector_translation, inliers = cv2.solvePnPRansac(corrs, image_points , gt_intrinsic, np.array([[0.0, 0.0, 0.0, 0.0]]), reprojectionError=50, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
        r = RR.from_rotvec(vector_rotation[..., 0])
        pred_R = r.as_matrix()
        return corr_entry, corr_entry_std, corr_exit, corr_exit_std, pred_R, vector_translation
