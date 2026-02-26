#!/usr/bin/python
# -*- encoding: utf-8 -*-
#-****************************************************************-#
# Author: Song
#-****************************************************************-#

import sys
sys.path.append("..")
#import numpy as np
from decimal import Decimal
import datetime
import time
import os
#import cv2
import torch
import kornia
import torch.backends.cudnn
import torchvision.transforms
import torch.utils.data

from dataloader.fusion_dataloader import FuseTrainData, FuseTestData
from models.fsfnet import FSFNet
from loss import fusion_loss
from models.mos_utils  import get_final_image , get_refined_image

import pathlib
import logging
import argparse

import warnings
import torch, gc
import os
import csv

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3000"
gc.collect()

torch.cuda.empty_cache()

warnings.filterwarnings('ignore')
print(os.getcwd())


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    
def hyper_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs',     type=int,   default=100,   help='epoch number')
    parser.add_argument('--lr',          type=float, default=1e-6,  help='learning rate')
    parser.add_argument('--lr_sod',      type=float, default=5e-5,  help='learning rate for SOD')
    parser.add_argument('--batchsize',   type=int,   default=4,    help='training batch size')
    parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
    parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
    parser.add_argument('--lw',          type=float, default=0.001, help='weight')
    parser.add_argument('--lr_decay',    type=float, default=0.97,   help='decay rate of learning rate')
    parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,   default=60,    help='every n epochs decay learning rate')
    parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
    parser.add_argument("--cuda",        action="store_false", help="Use cuda?")
    parser.add_argument('--gpu_id',      type=str,   default='1',   help='train use gpu')

    parser.add_argument('--alternate',   type=int,   default=6,   help='alternate training times')
    parser.add_argument('--checkpoint',  type=str,   default='../checkpoint/[1,1,15,5,5]/',   help='save path of model')

    parser.add_argument('--rgb_root',      type=str, default='../Datesets/RGBT/VT5000/Train/RGB/',       help='the training rgb images root')
    parser.add_argument('--thermal_root',  type=str, default='../Datesets/RGBT/VT5000/Train/T/',         help='the training depth images root')
    parser.add_argument('--rgb_map',       type=str, default='../Datesets/RGBT/VT5000/Train/RGB_map_soft/', help='the training gt images root')
    parser.add_argument('--thermal_map',   type=str, default='../Datesets/RGBT/VT5000/Train/T_map_soft/',   help='the training gt images root')
    parser.add_argument('--val_rgb_root',        type=str, default='../Datesets/RGBT/VT5000/Test/RGB/',      help='the test rgb images root')
    parser.add_argument('--val_thermal_root',    type=str, default='../Datesets/RGBT/VT5000/Test/T/',    help='the test depth images root')
    #parser.add_argument('--val_gt_root',         type=str, default='../Datesets/RGBT/VT5000/Test/GT/',       help='the test gt images root')
    #parser.add_argument('--fused_test_dir',      type=str, default='../Results/Fusion/RoadScene-master/Fused_2024_5_6/',    help='the path to save fused images')
    #parser.add_argument('--cos_test_dir',      type=str, default='../Results/cos/RoadScene-master/Fused_2024_5_27/',    help='the path to save cos images')
    opt = parser.parse_args()
    return opt

def create_logger(filepath):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    console_logger = logging.getLogger('ConsoleLogger')
    file_logger = logging.getLogger('FileLogger')

    file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr_current = opt.lr * opt.lr_decay ** (epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current



def Train_FSFNet(opt, file_logger):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    '''Defining ckpt, dataloader, model, optimizer, loss functions for Fusion phase'''
    checkpoint = '../checkpoint/fusion_celue/'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    # Loading datasets
    path_rgb = pathlib.Path(opt.rgb_root)
    path_thermal = pathlib.Path(opt.thermal_root)
    path_rgb_map = pathlib.Path(opt.rgb_map)
    path_thermal_map = pathlib.Path(opt.thermal_map)

    crop = torchvision.transforms.RandomResizedCrop(352)
    data = FuseTrainData(path_thermal, path_rgb, path_thermal_map, path_rgb_map, crop)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=opt.batchsize,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=False)
    file_logger.info("the training dataset is length:{}".format(len(train_loader)))
    train_loader.n_iter = len(train_loader)

    # Defining network, optimizer and loss functions
    fusionNet = FSFNet(64).cuda()
    optimizer = torch.optim.Adam(params=fusionNet.parameters(), lr=opt.lr)
    criteria_fusion = fusion_loss.FusionLoss().to(device)

    # Initialize a list to store the loss values
    loss_values = []

    # Ensure the results folder exists
    os.makedirs('../results', exist_ok=True)
    start_time = time.time()

    # Starting training
    nEpochs = 30
    st = glob_st = time.time()
    for epoch in range(opt.start_epoch, nEpochs + 1):
        fusionNet.train()
        for it, ((T, YCrCb), (T_path, RGB_path), (T_map, RGB_map)) in enumerate(train_loader):
            img_t = T.cuda()
            img_ycrcb = YCrCb.cuda()
            map_t = T_map.cuda()
            map_rgb = RGB_map.cuda()

            img_t_3 = torch.cat((img_t, img_t, img_t), dim=1)
            img_y_1 = img_ycrcb[:, :1, :, :]

            orig_transmission_map_x, orig_atmosphere_light_x, orig_refined_image_x = get_refined_image(img_ycrcb)

            fus_y , x_free, y_free , cos , pred_transmission_map_x, pred_atmosphere_light_x = fusionNet(img_ycrcb, img_t_3)
            fusion_ycrcb = torch.cat((torch.clamp(fus_y, 0, 1), img_ycrcb[:, 2:, :, :], img_ycrcb[:, 1:2, :, :]), dim=1)
            fusion_image = kornia.color.ycbcr_to_rgb(fusion_ycrcb)

            # Compute loss
            loss_fusion, loss_in, loss_grad , ms_ssim_loss  , tm_l1 , atmos_l1 = criteria_fusion(fus_y, img_t, img_y_1, map_t, map_rgb , orig_transmission_map_x , orig_atmosphere_light_x , pred_transmission_map_x , pred_atmosphere_light_x)

            loss = loss_fusion

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusionNet.parameters(), max_norm=1.0)
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * (epoch - 1) + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))


            # Logging for every 10 steps
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'Epoch: {epoch}',
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ms_ssim_loss: {ms_ssim_loss:.4f}',
                        'tm_l1: {tm_l1:.4f}',
                        'atmos_l1: {atmos_l1:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    epoch=epoch,
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    ms_ssim_loss=ms_ssim_loss.item(),
                    tm_l1=tm_l1.item(),
                    atmos_l1=atmos_l1.item(),
                    time=t_intv,
                    eta=eta,
                )
                file_logger.info(msg)
                loss_values.append([epoch, it, loss_fusion.item(), loss_in.item(), loss_grad.item(), ms_ssim_loss.item(), tm_l1.item(), atmos_l1.item()])
                # Print the loss values array after each update
                print(f"Loss values at Epoch {epoch}, Iteration {it}: {loss_values[-1]}")
                st = ed
                       
        # Save model checkpoint
        if 25 <= epoch <= 30:
            current_model_file = os.path.join(checkpoint, f'fusion_model_epoch_{epoch}.pth')
            torch.save(fusionNet.state_dict(), current_model_file)
            current_optim_file = os.path.join(checkpoint, f'optimizer_epoch_{epoch}.pt')
            torch.save(optimizer.state_dict(), current_optim_file)
            print(f"Model and optimizer for epoch {epoch} saved.")
            
        # Record the training end time
    end_time = time.time()
    
    # Calculate the total training time
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    print(f"Total training time: {total_time_str}")

    print("Training complete.")

    # Save the loss values to a CSV file after training is complete
    with open('../results/loss_values.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Iteration', 'Loss_Fusion', 'Loss_In', 'Loss_Grad', 'MS_SSIM_Loss', 'TM_L1', 'Atmos_L1'])
        writer.writerows(loss_values)


torch.cuda.empty_cache()

if __name__ == "__main__":
    opt = hyper_args()
    console_logger, file_logger = create_logger('../log/train_fsfnet_0912.log/')
    Train_FSFNet(opt, file_logger)
 