import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms_tv
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.evaluations import calc_mTRE

def model_parser(model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
    base_model = None

    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight, dropout_rate)
    elif model == 'Resnet':
        base_model = models.resnet34(pretrained=False)
        out_channels = base_model.conv1.out_channels
        kernel_size = base_model.conv1.kernel_size
        padding = base_model.conv1.padding
        stride = base_model.conv1.stride
        base_model.conv1 = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        network = ResNet(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'ResnetSimple':
        base_model = models.resnet34(pretrained=True)
        network = ResNetSimple(base_model, fixed_weight)
    else:
        assert 'Unvalid Model'

    return network


class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25
            
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        target_q = F.normalize(target_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

            
        loss = torch.exp(-self.sx)*loss_x \
               + self.sx \
               + torch.exp(-self.sq)*loss_q \
               + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()


class ResNet(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(ResNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class ResNetSimple(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNetSimple, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, 4, bias=False)

        init_modules = [self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation

class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate =dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.ori2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori


class LitPoseNet(pl.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = model_parser("Resnet")
        self.criterion = PoseLoss(self.device, learn_beta=False)
        self.lr = model_cfg["lr"]
        self.transforms = transforms_tv.Compose([
            transforms_tv.RandomChoice(
            [
                transforms_tv.ColorJitter(brightness=1, contrast=1),
                transforms_tv.RandomInvert()
            ]
            ),
            transforms_tv.RandomErasing()
        ]
        )

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        imgs = self.transforms(imgs)
        extrinsic = batch["extrinsic"]
        pos = extrinsic[:, :3, 3]
        ori = transforms.matrix_to_quaternion(extrinsic[:, :3, :3])
        pred_pos, pred_ori, _ = self.forward(imgs)
        loss, loss_x, loss_q = self.criterion(pred_pos, pred_ori, pos, ori)
        self.log('train/loss', loss.item())
        self.log('train/translation', loss_x)
        self.log('train/rotation', loss_q)
        return {"loss": loss, "pred_pos": pred_pos, "pred_ori": pred_ori}

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        extrinsic = batch["extrinsic"]

        pos = extrinsic[:, :3, 3]
        ori = transforms.matrix_to_quaternion(extrinsic[:, :3, :3])
        pred_pos, pred_ori, _ = self.forward(imgs)
        loss, _, _ = self.criterion(pred_pos, pred_ori, pos, ori)
        if batch_idx == 0:
            for sample_idx in range(imgs.shape[0]):
                gt_extrinsic = batch["extrinsic"][sample_idx][:-1, :].cpu().numpy()
                vol_landmarks = batch["vol-landmarks"][sample_idx].cpu().numpy()
                
                pred_extrinsic = np.concatenate([transforms.quaternion_to_matrix(pred_ori[sample_idx]).cpu().numpy(), pred_pos[sample_idx][..., None].cpu().numpy()], axis=-1)
                mTRE = calc_mTRE(vol_landmarks, pred_extrinsic=pred_extrinsic, gt_extrinsic=gt_extrinsic)
                self.log("val/mTRE", mTRE)

        self.log('val/loss', loss.item())
        return {"loss": loss, "pred_pos": pred_pos, "pred_ori": pred_ori}
    
    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
