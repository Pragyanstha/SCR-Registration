import torch
from torch import nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as RR

from utils.evaluations import calc_mTRE
from utils.model_helpers import heatmap2location, ncc_2d
from models.unet import UNet


class LitDFLNet(pl.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.DFLNet = UNet(1, 14)
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
        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.DFLNet(x)
    
    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        imgs = self.transforms(imgs)
        gt_heatmaps = batch["heatmaps"]
        pred_heatmaps = self.forward(imgs).unsqueeze(-3)
        # loss  = (1.0-ncc_2d(pred_heatmaps, gt_heatmaps)).mean()
        loss = ((ncc_2d(pred_heatmaps, gt_heatmaps) - 1)* -0.5).mean()
        # loss = self.mse(pred_heatmaps, gt_heatmaps)
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_heatmaps = batch["heatmaps"]
        pred_heatmaps = self.forward(imgs).unsqueeze(-3)

        loss = ((ncc_2d(pred_heatmaps, gt_heatmaps) - 1)* -0.5).mean()

        if batch_idx == 0:
            for sample_idx in range(imgs.shape[0]):
                intrinsic = batch["intrinsic"][sample_idx].cpu().numpy()
                gt_extrinsic = batch["extrinsic"][sample_idx][:-1, :].cpu().numpy()
                pred_heatmap = pred_heatmaps[sample_idx].cpu()
                pred_proj_landmarks = heatmap2location(pred_heatmap)
                pred_heatmap = pred_heatmap.numpy()
                vol_landmarks = batch["vol-landmarks"][sample_idx].cpu().numpy()
                selections =  ~np.any(np.isinf(pred_proj_landmarks), axis=-1)
                try:
                    success, vector_rotation, vector_translation, inliers = cv2.solvePnPRansac(
                        vol_landmarks[selections], pred_proj_landmarks[selections] , intrinsic, np.array([[0.0, 0.0, 0.0, 0.0]]),
                        reprojectionError=20, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)
                except Exception as e:
                    continue

                r = RR.from_rotvec(vector_rotation[..., 0])
                pred_extrinsic = np.concatenate([r.as_matrix(), vector_translation], axis=-1)
                mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
                self.log("val/mTRE", mTRE)
            return {
                "loss": loss.item(),
                "img": imgs[0],
                "pred": pred_heatmaps[0].sum(dim=0),
                "gt": gt_heatmaps[0].sum(dim=0)
            }

        self.log("val/loss", loss)
        return {
            "loss": loss.item()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                "monitor": "val/loss"
                }


    def validation_epoch_end(self, batch):
        img = batch[0]["img"]
        pred = batch[0]["pred"]
        gt = batch[0]["gt"]
        self.logger.log_image(key="samples", images=[img, pred, gt])
