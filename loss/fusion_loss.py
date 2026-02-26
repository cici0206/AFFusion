import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.ms_ssim import MSSSIM

class FusionLoss(nn.Module):
    def __init__(self, alpha=5, beta=20, theta=5 , x=1 , y=1):  
        super(FusionLoss, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.grad_loss = JointGrad()
        
        self.max_loss = nn.L1Loss()
        self.mask_loss = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.x = x
        self.tm_l1 = nn.L1Loss()
        self.atmos_l1 = nn.L1Loss()
        
        self.y = y

    def forward(self, im_fus, im_ir, im_vi, map_ir, map_vi , orig_tm ,orig_atmos , pred_tm , pred_atmos ):
        ms_ssim_loss = (1 - self.ms_ssim(im_fus, (0.8 * map_ir * im_ir + 1.2 * map_vi * im_vi)))
        l1_loss = self.l1_loss(im_fus, (0.8 * map_ir * im_ir + 1.2 * map_vi * im_vi))
        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)
        tm_l1 = self.tm_l1(orig_tm, pred_tm)
        atmos_l1 = self.atmos_l1(orig_atmos, pred_atmos)
        
        fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss + self.x * tm_l1 + self.y * atmos_l1
        return fuse_loss, l1_loss, grad_loss , ms_ssim_loss ,tm_l1 , atmos_l1



class MaxFusionloss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MaxFusionloss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        generate_img_y = generate_img[:, :1, :, :]

        return self.loss_weight * F.l1_loss(generate_img_y, torch.max(image_y, image_ir))
        
        
        

class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = torch.abs(self.laplacian(im_fus, 3))

        loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        # loss_JGrad = 0.8 * self.l1_loss(ir_grad, fus_grad) + 0.2 * self.l1_loss(vi_grad, fus_grad)

        return loss_JGrad


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss