# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:41:09 2023

@author: SimethJ
"""
import pathlib
import torch


import pandas as pd

from util.visualizer import visualize_train_case_slices, visualize_val_case_slices

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:45:59 2023

@author: SimethJ

Updated inference function that will process all the nifti files in a specified
folder (default: )

assumes nifti files with units 1e-3 mm^2/s

"""

#import nibabel as nib
import torch as t
from torch.cuda.amp import GradScaler, autocast
import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms

#import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import torch.multiprocessing

import torch.utils.data
from options.train_options import TrainOptions
from models.models import create_model
# from util import util
from PCA_DIL_inference_utils import sliding_window_inference 

opt = TrainOptions().parse()
opt.isTrain=True

# from pathlib import Path
from glob import glob
import datetime

from typing import Tuple, Optional, Dict
import monai
from monai.handlers.utils import from_engine
from monai.data import DataLoader, create_test_image_3d, MetaTensor
from monai.data import decollate_batch


#from monai.inferers import SliceInferer
#from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    Spacingd,
    Orientationd,
    SqueezeDimd,
    ResizeWithPadOrCropd,
    CropForegroundd,
    RandRotated,
    CenterSpatialCropd,
    Transposed,
    ResampleToMatchd,
    RandSpatialCropd,
    Invertd,
    AsDiscreted,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandBiasFieldd,
    RandFlipd,
    RandAxisFlipd,
)





def copy_info(src, dst):

    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())

    return dst

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)




def scale_ADC(image):
    adc_max=image.data.amax()
    adc_max=adc_max.data.cpu().numpy()
    
    
    if adc_max<0.05:# input likely in  mm^2/s, 
        print('Confirm units of ADC: input probably incorrectly in mm^2/s, normalizing with that assumption')
        multiplier=1e6
    elif adc_max<50:#input likely in  1e-3 mm^2/s, 
        print('Confirm units of ADC: input probably incorrectly in 1e-3  mm^2/s, normalizing with that assumption')
        multiplier=1
    elif adc_max<50000:#input likely in  1e-6 mm^2/s, 
        multiplier=1e-3
        #print('Confirm units of ADC: input probably incorrectly in 1e-6  mm^2/s, normalizing with that assumption')
    else:
        print('Confirm units of ADC: values too large to be 1e-6  mm^2/s')
    adc=2*(image.data*multiplier)/3.5-1
    adc=t.clip(adc,min=-1.0,max=1.0)
    
    return adc




def find_file_with_string(folder_path, target_string):
    # Get the list of files in the specified folder
    files = os.listdir(folder_path)
    # Iterate through the files and find the first file containing the target string
    for file in files:
        if target_string in file:
            # If the target string is found in the file name, return the file
            return file
    # If no matching file is found, return None
    return None


def update_metrics_over_time_dict(dice_over_time: Dict, current_iteration_data: Dict, file_suffix: str = None) -> Dict:
    epoch = current_iteration_data.get('epoch')
    lr = current_iteration_data.get('lr')
    errors = current_iteration_data.get('errors')
    start_training = current_iteration_data.get('epoch_zero_datetime')
    avg_dice = current_iteration_data.get('avg_dice')
    median_dice = current_iteration_data.get('median_dice')
    dices = current_iteration_data.get('dices')

    d0 = errors.get("d0")
    seg_loss = errors.get('Seg_loss')
    seg_loss_item = seg_loss.item() if isinstance(seg_loss, torch.Tensor) else None

    metrics_data = [epoch, lr, seg_loss_item, d0, avg_dice, median_dice]
    index_data = ['epoch', 'lr',  'seg_loss', 'd0', 'avg_dice', 'median_dice']

    def generate_dice_values_labels(dices):
        """For each dice in dices, create a data list with the dices for each case and a index_data list indexing the dice to a case"""
        dice_values = []
        dice_labels = []
        for count, dice in enumerate(dices):
            dice_values.append(dice)
            dice_label = f'case_{count}_dice'
            dice_labels.append(dice_label)
        return dice_values, dice_labels

    # roll out dice values and generate labels to be added to the metrics and index data
    if dices:
        dice_values, dice_labels = generate_dice_values_labels(dices)
        metrics_data.extend(dice_values)
        index_data.extend(dice_labels)

    dice_over_time[f'{datetime.datetime.now()}'] = metrics_data
    dice_df = pd.DataFrame(dice_over_time, index=index_data)

    filename = f"metrics_over_time_{start_training}_{file_suffix}.xlsx" if (
        file_suffix) else f"metrics_over_time_{start_training}.xlsx"
    dice_df.to_excel(filename)
    print(f'Updated excel at: "{filename}"')
    return dice_over_time

# def update_metrics_over_time(dice_over_time:Dict, epoch, lr, errors,
#                              start_training, avg_dice=None,file_suffix:str= None) -> Dict:
#     if avg_dice:
#         metrics_data = [epoch, lr, avg_dice,errors['Seg_loss'].item(), errors['d0']]
#         index_data = ['epoch', 'lr', 'avg_dice', 'seg_loss', 'd0']
#     else:
#         metrics_data =[epoch, lr, errors['Seg_loss'].item(), errors['d0']]
#         index_data = ['epoch', 'lr', 'seg_loss', 'd0']
#
#     dice_over_time[f'{datetime.datetime.now()}'] = metrics_data
#     dice_df = pd.DataFrame(dice_over_time, index=index_data)
#
#     filename = f"metrics_over_time_{start_training}_{file_suffix}.xlsx" if (
#         file_suffix) else f"metrics_over_time_{start_training}.xlsx"
#     dice_df.to_excel(filename)
#     print(f'Updated excel at: "{filename}"')
#     return dice_over_time

def get_case_dice_2D_temp2(seg_data,label_data,smooth):
    # print("seg length: ", len(seg))
    seg = seg_data[0]
    # print("seg shape: ", np.shape(seg))
    seg = np.array(seg)
    seg = np.squeeze(seg)
    seg[seg >= 0.5] = 1.0
    seg[seg < 0.5] = 0.0

    # print(np.ndim(seg))
    # print(seg)
    # print(np.shape(seg))

    # t2w=t2w[0,:,:,:,2]
    # adc=adc[0,:,:,:,2]
    # label_val=label_val[0,:,:,:,2]

    # transform = transforms.ToPILImage()
    # t2w = transform(t2w)
    # adc = transform(adc+1)
    # label_png = transform(label_val)
    # t2w.save('t2w.png')
    # adc.save('adc.png')
    # label_png.save('label.png')

    seg_temp = np.array(seg)
    gt_temp = np.array(label_data.cpu())

    seg_flt = seg_temp.flatten()
    gt_flt = gt_temp.flatten()

    intersection = np.sum(seg_flt * (gt_flt > 0))
    val_dice_2D_temp = (2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
    print(val_dice_2D_temp)
    return val_dice_2D_temp, seg_temp



def get_case_dice_2D_temp(seg_data, label_data, smooth=1e-6):
    """
    Optimized 2D Dice score calculation using PyTorch operations on GPU.

    Args:
        seg_data: Predicted segmentation tensor (on GPU)
        label_data: Ground truth label tensor (on GPU)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        tuple: (dice_score, processed_segmentation)
    """
    # Keep data on GPU and avoid unnecessary transfers
    seg = seg_data[0]  # Assuming batch size of 1

    # Squeeze
    seg = torch.squeeze(seg)

    # Binary thresholding
    seg = (seg >= 0.5).float()

    # Ensure label_data is on the same device as seg
    label_data = label_data.to(seg.device)

    # Flatten tensors using view
    seg_flat = seg.view(-1)
    gt_flat = (label_data > 0).float().view(-1)

    # Calculate intersection and dice score
    intersection = torch.sum(seg_flat * gt_flat)
    dice_score = (2. * intersection + smooth) / (torch.sum(seg_flat) + torch.sum(gt_flat) + smooth)

    # Only convert to CPU at the end if necessary
    # dice_score_val = dice_score.item()
    # print(dice_score_val)

    return dice_score, seg


def train():
    opt = TrainOptions().parse()
    opt.isTrain = True
    mr_paths = []
    seg_paths = []

    #Define input dimensions and resolution for inference model
    PD_in=np.array([0.6250, 0.6250, 3]) # millimeters
    DIM_in=np.array([128,128,opt.nslices]) # 128x128 5 Slices
    # TODO: Check here if amount of wanted channels is used
    nmodalities=2

    root_dir=os.getcwd()
    # opt.nchannels=opt.nslices*nmodalities

    model = create_model(opt)

    model_path = os.path.join(root_dir,'deep_model','MRRNDS_model') #load weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_MR_seg_A(model_path) #use weights
    # model.to(device)

    #freeze all layers
    # for param in model.netSeg_A.parameters():
    #         param.requires_grad = False


    #unfreeze final layer and Channel input block for finetuning
    # model.netSeg_A.CNN_block1.requires_grad_(True)
    # model.netSeg_A.out_conv.requires_grad_(True)
    # model.netSeg_A.out_conv.requires_grad_(True)

    #we can unfreeze other layers either right away or on a epoch based schedule

    #we likely want to train a little bit just to get the channel adaptation working
    #then if we have a lot of data unfreeze more layers as below:

    # #first 5 layers

    # model.netSeg_A.CNN_block2.requires_grad_(True)
    # model.netSeg_A.CNN_block3.requires_grad_(True)
    # model.netSeg_A.RU1.requires_grad_(True)
    # model.netSeg_A.RU2.requires_grad_(True)
    # model.netSeg_A.RU3.requires_grad_(True)

    # #last 3 layers (not counting output conv)
    # model.netSeg_A.RU33.requires_grad_(True)
    # model.netSeg_A.RU22.requires_grad_(True)
    # model.netSeg_A.RU11.requires_grad_(True)


    # directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # if directory is not None:
    #     os.makedirs(directory, exist_ok=True)
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    # print(root_dir)

    wt_path = root_dir + 'ct_seg_val_loss.csv'

    fd_results = open(wt_path, 'w')
    fd_results.write('train loss, seg accuracy,\n')

    #get nii and seg data
    #nmodalities=2
    #impath = os.path.join(path,'nii_vols') #load weights
    # impath = os.path.join(root_dir,'training_data') #load weights
    # valpath = os.path.join(root_dir,'validation_data') #load weights

    # images = sorted(glob(os.path.join(root_dir, "adc*.nii.gz")))
    # train_images = sorted(glob(os.path.join(impath, "*_ep2d_diff_*.nii.gz"))) #adc keywords from filename
    #segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
    # train_segs = sorted(glob(os.path.join(impath, "*t2_tse_tra*_ROI.nii.gz")))
    # images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
    # train_images_t2w = sorted(glob(os.path.join(impath, "*_t2_tse*.nii.gz"))) #t2w keywords from filename

    # val_images = sorted(glob(os.path.join(valpath, "*_ep2d_diff_*.nii.gz"))) #adc keywords from filename
    #segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
    # val_segs = sorted(glob(os.path.join(valpath, "*t2_tse_tra*_ROI.nii.gz")))
    # images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
    # val_images_t2w = sorted(glob(os.path.join(valpath, "*_t2_tse*.nii.gz"))) #t2w keywords from filename

    # NKI data
    impath = os.path.join(root_dir,'training_data\\nki_resampled\\train') #load data
    valpath = os.path.join(root_dir,'training_data\\nki_resampled\\val') #load weights

    train_images = sorted(glob(os.path.join(impath, "*adc*.nii")))
    train_segs = sorted(glob(os.path.join(impath, "*LES*.nii")))
    train_images_t2w = sorted(glob(os.path.join(impath, "*tt2*.nii"))) #t2w keywords from filename

    val_images = sorted(glob(os.path.join(valpath, "*adc*.nii")))
    val_segs = sorted(glob(os.path.join(valpath, "*LES*.nii")))
    val_images_t2w = sorted(glob(os.path.join(valpath, "*tt2*.nii"))) #t2w keywords from filename


    # print(images)
    # print(segs)
    # print(images_t2w)
    #can add additional modalities with thier keyname
    #images_ktrans = sorted(glob(os.path.join(root_dir, "ktrans*.nii.gz")))

    # print(val_images)
    # print(val_segs)
    # print(val_images_t2w)


    # n_val=3
    # n_train=7
    train_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(train_images, train_segs, train_images_t2w)] #first n_train to training
    val_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(val_images, val_segs, val_images_t2w)] #last  n_val to validation

    # total_dataset = 10
    # n_val = round(total_dataset * 0.3)
    # n_train = total_dataset - n_val
    #
    # train_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in
    #                zip(images[:n_train], segs[:n_train], images_t2w[:n_train])]  # first n_train to training
    # val_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in
    #              zip(images[-n_val:], segs[-n_val:], images_t2w[-n_val:])]  # last  n_val to validation

    # print(train_files)
    # print(val_files)

    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "t2w", "seg"]),
            EnsureChannelFirstd(keys=["img", "t2w", "seg"]),
            Orientationd(keys=["img", "t2w", "seg"], axcodes="RAS"),

            ResampleToMatchd(keys=["img"],
                             key_dst="t2w",
                             mode="bilinear"),
            ResampleToMatchd(keys=["seg"],
                             key_dst="t2w",
                             mode="nearest"),
            Spacingd(keys=["img", "t2w", "seg"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode=("bilinear", "bilinear", "nearest")),
            # RandRotate90d(keys=["img", "t2w", "seg"], prob=0.4, max_k=1),
            RandFlipd(keys=["img", "t2w", "seg"], prob=0.5),
            # RandAxisFlipd(keys=["img", "t2w", "seg"], prob=0.7),

            # ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
            RandAdjustContrastd(keys=["img"], prob=0.05),
            RandHistogramShiftd(keys=["img"], prob=0.05),
            RandBiasFieldd(keys=["img"], prob=0.05),
            RandAdjustContrastd(keys=["t2w"], prob=0.05),
            RandHistogramShiftd(keys=["t2w"], prob=0.05),
            RandBiasFieldd(keys=["t2w"], prob=0.05),

            ScaleIntensityRangePercentilesd(keys=["img", "t2w"], lower=0, upper=98, b_min=-1.0, b_max=1.0, clip=True),
            # RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
            # CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
            # ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
            # RandRotate90d(keys=["img","t2w","seg"], prob=0.2, spatial_axes=[0, 1]),
            # RandRotate90d(keys=["img","t2w","seg"], prob=0.2, spatial_axes=[1, 2]),
            RandGaussianNoised(keys=["img", "t2w"], prob=0.10),
            RandGaussianSmoothd(keys=["img", "t2w"], prob=0.10),
            # CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),

            # CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[-1,-1,opt.extra_neg_slices+(opt.nslices-1)/2]),
            CropForegroundd(keys=["img", "t2w", "seg"], source_key="seg",
                            margin=[96, 96, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            RandRotated(keys=["img", "t2w", "seg"], prob=1.0, range_x=1, range_y=0.0, range_z=0.0, mode=("bilinear", "bilinear", "nearest")),
            # RandRotated(keys=["img", "t2w", "seg"], prob=0.6, range_x=0, range_y=0.1, range_z=0, mode=("bilinear", "bilinear", "nearest")),#add pitch
            # RandRotated(keys=["img", "t2w", "seg"], prob=0.6, range_x=0, range_y=0, range_z=0.1, mode=("bilinear", "bilinear", "nearest")),#add yaw
            CenterSpatialCropd(keys=["img", "t2w", "seg"], roi_size=[200, 200, -1]),
            RandSpatialCropd(keys=["img", "t2w", "seg"], roi_size=(128, 128, opt.nslices), random_size=False),
            Transposed(keys=["img", "seg", "t2w"], indices=[3, 2, 1, 0]),
            SqueezeDimd(keys=["img", "t2w", "seg"], dim=-1),
            EnsureTyped(keys=["img", "t2w", "seg"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "t2w", "seg"]),
            EnsureChannelFirstd(keys=["img", "t2w", "seg"]),
            Orientationd(keys=["img", "t2w", "seg"], axcodes="RAS"),

            ResampleToMatchd(keys=["img"],
                             key_dst="t2w",
                             mode="bilinear"),
            ResampleToMatchd(keys=["seg"],
                             key_dst="t2w",
                             mode="nearest"),
            Spacingd(keys=["img", "t2w"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode="bilinear"),
            Spacingd(keys=["seg"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode="nearest"),

            ScaleIntensityRangePercentilesd(keys=["img", "t2w"], lower=0, upper=98, b_min=-1.0, b_max=1.0, clip=True),
            # ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
            # RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
            # CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
            # ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
            # RandRotate90d(keys=["img","t2w","seg"], prob=0.5, spatial_axes=[0, 1]),
            # RandRotated(keys=["img","t2w","seg"], prob=0.9, range_x=3.0),
            # CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),
            CropForegroundd(keys=["img", "t2w", "seg"], source_key="seg",
                            margin=[96, 96, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            # RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),
            EnsureTyped(keys=["img", "t2w", "seg"]),
        ]
    )

    # post_transforms = Compose([
    #         EnsureTyped(keys="pred"),
    #         Invertd(
    #             keys="pred",
    #             transform=val_transforms,
    #             orig_keys="t2w",
    #             meta_keys="pred_meta_dict",
    #             orig_meta_keys="image_meta_dict",
    #             meta_key_postfix="meta_dict",
    #             nearest_interp=True,
    #             to_tensor=True,
    #         ),
    #         AsDiscreted(keys="pred", argmax=False),
    #
    #     ])


    # 3D dataset with preprocessing transforms
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)

    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)


    train_loader = DataLoader(
        train_ds,
        # batch_size=1,
        batch_size=opt.batchSize,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        # drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    check_data = monai.utils.misc.first(train_loader)
    print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

    num_epochs=opt.niter+opt.niter_decay
    epoch_loss_values = []
    best_dice=0
    mixup_ab=opt.mixup_betadist
    metrics_over_time = {'train':{} , 'val': {}}
    # Note datetime of epoch 0
    epoch_zero_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Started training at {epoch_zero_datetime}")
    for epoch in range(num_epochs+1):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(f'Current epoch: {epoch}, current datetime: {current_datetime}')
        lr=model.get_curr_lr()
        # if epoch==100:
        #     model.netSeg_A.out_conv.requires_grad_(True)
        # if epoch==round(num_epochs*opt.unfreeze_fraction1):
        #     print('  >>unfreeze layer 2 and RU11')
        #     model.netSeg_A.CNN_block2.requires_grad_(True)
        #     model.netSeg_A.RU11.requires_grad_(True)
        # if epoch==round(num_epochs*opt.unfreeze_fraction2):
        #     print('  >>unfreeze layer 3, RU1,RU2,RU3, RU33 and RU22')
        #     model.netSeg_A.CNN_block2.requires_grad_(True)
        #     model.netSeg_A.CNN_block3.requires_grad_(True)
        #     model.netSeg_A.RU1.requires_grad_(True)
        #     model.netSeg_A.RU2.requires_grad_(True)
        #     model.netSeg_A.RU3.requires_grad_(True)

            #last 3 layers (not counting output conv)
            # model.netSeg_A.RU33.requires_grad_(True)
            # model.netSeg_A.RU22.requires_grad_(True)
            # model.netSeg_A.RU11.requires_grad_(True)
        #if epoch==round(num_epochs*0.9):
        # print("-" * 10)
        # print(f"epoch {epoch + 1}/{num_epochs}")
        epoch_loss, train_step = 0, 0
        train_dices =[]
        train_label_slice_num = np.int32((opt.nslices-1)/2)
        for batch_data in train_loader:
            train_step += 1

            # img_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            # print(img_name)

            adc, labels, t2w = batch_data["img"].to(device, non_blocking=True), batch_data["seg"].to(device, non_blocking=True), batch_data["t2w"].to(device, non_blocking=True)
            # img_name=t2w.meta['filename_or_obj']
            # print(img_name)

            # if torch.sum(labels)>0:
            #     adc=scale_ADC(adc)
            # TODO: Check here if amount of wanted channels is used
            inputs=torch.cat((adc,t2w),dim=1)
            labels=labels[:,train_label_slice_num,:,:]

            labels = torch.clamp(labels, 0.001, 0.999).float()
            # inputs, labels = mixup(inputs, labels, np.random.beta(mixup_ab, mixup_ab))

            model.set_input_sep(inputs,labels)
            model.optimize_parameters()
            if (epoch % (num_epochs/2)) == 0:
                train_vis_folder = pathlib.Path(
                f"D:\Projects\PCA_Segmentation_MRRN_training\MRRN_PCA_Training\\visualisation\mixup\\nki_resampled\\train\{epoch_zero_datetime}")
                visualize_train_case_slices(image_data=[adc, t2w],
                                            contour_data=[labels],
                                            contour_colors=['r'],
                                            # Assuming red for ground truth and blue for prediction
                                            contour_labels=['Ground Truth'],
                                            save_folder=train_vis_folder,
                                            train_label_slice_index=train_label_slice_num,
                                            additional_info={
                                          'Epoch': epoch,
                                          'Input Data': str(train_loader.dataset.data[train_step - 1])
                                      },
                                            case_id=None  # Or provide a specific case ID if available
                                            )


        if (epoch % opt.display_freq) == 0:
            #                     # save_result = total_steps % opt.update_html_freq == 0
            #                     # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            errors = model.get_current_errors()['Seg_loss']

            # message = '(epoch: %d) ' %epoch
            # for k, v in errors.items():
            #     message += '%s: %.3f ' % (k, v)
            #     print(message)

            # t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        train_lr = model.get_curr_lr()
        train_errors = model.get_current_errors()
        # train_avg_dice_2D = np.average(train_dices)
        if (epoch%opt.display_freq)==0:
            current_iteration_train_data = {"epoch": epoch, "lr": train_lr, "errors": train_errors,
                                          "epoch_zero_datetime": epoch_zero_datetime,
                                          }
            metrics_over_time['train'] = update_metrics_over_time_dict(metrics_over_time['train'],
                                                                     current_iteration_train_data
                                                                     , 'train')
            print("-" * 10)
            print("-" * 10)
            print(f"Current epoch {epoch}/{num_epochs}")
            print("-" * 10)
            print("-" * 10)

            with torch.no_grad(): # no grade calculation
                val_dices=[]
                smooth=1
                val_step=0
                for val_data in val_loader:
                    val_step += 1
                    adc, label_val,t2w = val_data["img"].to(device, non_blocking=True), val_data["seg"].to(device, non_blocking=True), val_data["t2w"].to(device, non_blocking=True)


                    label_val_vol=label_val
                    #if torch.sum(label_val)>0:
                    # adc=scale_ADC(adc)

                    val_inputs=torch.cat((adc,t2w),dim=1)


                    with autocast(enabled=True):
                        #pass model segmentor and region info
                        #input, roi size, sw batch size, model, overlap (0.5)
                        val_data["pred"] = sliding_window_inference(val_inputs,
                                                                    (128, 128, opt.nslices),
                                                                    1,
                                                                    model.netSeg_A,
                                                                    overlap=0.66,
                                                                    mode="gaussian",
                                                                    sigma_scale=[0.128, 0.128,0.01])

                    # seg = from_engine(["pred"])(val_data)
                    #print("seg length: ", len(seg))
                    # seg = seg[0]
                    #print("seg shape: ", np.shape(seg))
                    # seg = np.array(seg)
                    # seg=np.squeeze(seg)
                    # seg[seg >= 0.5]=1.0
                    # seg[seg < 0.5]=0.0





                    # print(np.ndim(seg))
                    # print(seg)
                    # print(np.shape(seg))

                    # t2w=t2w[0,:,:,:,2]
                    # adc=adc[0,:,:,:,2]
                    # label_val=label_val[0,:,:,:,2]


                    # transform = transforms.ToPILImage()
                    # t2w = transform(t2w)
                    # adc = transform(adc+1)
                    # label_png = transform(label_val)
                    # t2w.save('t2w.png')
                    # adc.save('adc.png')
                    # label_png.save('label.png')




                    # seg_temp=np.array(seg)
                    # gt_temp=np.array(label_val_vol.cpu())
                    #
                    # seg_flt=seg_temp.flatten()
                    # gt_flt=gt_temp.flatten()
                    #
                    #
                    # intersection = np.sum(seg_flt * (gt_flt > 0))
                    # val_dice_2D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
                    # print(val_dice_2D_temp)
                    val_dice_2D_temp,seg_temp = get_case_dice_2D_temp(from_engine(["pred"])(val_data),label_val,smooth)

                    val_dices.append(val_dice_2D_temp)
                    if(epoch%(num_epochs//2))==0:
                        # visualize for inspection
                        val_vis_folder = pathlib.Path(
                            f"D:\Projects\PCA_Segmentation_MRRN_training\MRRN_PCA_Training\\visualisation\mixup\\nki_resampled\\val\{epoch_zero_datetime}")
                        visualize_val_case_slices(image_data=[adc, t2w],
                                                    contour_data=[label_val, seg_temp],
                                                    contour_colors=['r', 'b'],
                                                    # Assuming red for ground truth and blue for prediction
                                                    contour_labels=['Ground Truth', 'Prediction'],
                                                    save_folder=val_vis_folder,
                                                    additional_info={
                                                  'Dice': val_dice_2D_temp,
                                                  'Epoch': epoch,
                                                  'Input Data': str(val_loader.dataset.data[val_step - 1])
                                              },
                                                    case_id=None  # Or provide a specific case ID if available
                                                    )




                val_errors = model.get_current_errors()
                val_lr = model.get_curr_lr()
                val_avg_dice_2D = np.average(val_dices)
                val_median_dice_2D = np.median(val_dices)


                current_iteration_val_data = { "epoch":epoch,"lr": val_lr, "errors": val_errors,
                                              "epoch_zero_datetime": epoch_zero_datetime, "avg_dice": val_avg_dice_2D,
                                              "median_dice": val_median_dice_2D, "dices": val_dices}
                metrics_over_time['val'] = update_metrics_over_time_dict(metrics_over_time['val'],
                                                                         current_iteration_val_data
                                                                         ,'val')

                print('epoch %i' % epoch, 'DSC  %.2f' % val_avg_dice_2D, ' (best: %.2f)'  % best_dice)

                fd_results.write(str(val_avg_dice_2D) + '\n')
                fd_results.flush()

                if val_avg_dice_2D>best_dice:
                    print ('saving for Dice, %.2f' % val_avg_dice_2D, ' > %.2f' % best_dice)
                    im_iter=0
                    sitk.WriteImage(sitk.GetImageFromArray(t2w.cpu()), 'val%i_t2w.nii.gz' % im_iter)
                    sitk.WriteImage(sitk.GetImageFromArray(adc.cpu()), 'val%i_adc.nii.gz' % im_iter)
                    sitk.WriteImage(sitk.GetImageFromArray(label_val.cpu()), 'val%i_gtv.nii.gz' % im_iter)
                    sitk.WriteImage(sitk.GetImageFromArray(seg_temp.cpu()), 'val%i_seg.nii.gz' % im_iter)
                    # for vol_data in val_loader:
                    #     im_iter += 1
                    #     adc, label_val,t2w = vol_data["img"].to(device), vol_data["seg"].to(device), vol_data["t2w"].to(device)

                        # img_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
                        # print(img_name)

                        # label_val_vol=label_val
                        # if torch.sum(label_val)>0:
                        # adc=scale_ADC(adc)

                        # val_inputs=torch.cat((adc,t2w),dim=1)


                        # with autocast(enabled=True):
                            # pass model segmentor and region info
                            # input, roi size, sw batch size, model, overlap (0.5)
                            # vol_data["pred"] = sliding_window_inference(val_inputs,
                            #                                             (128, 128, 5),
                            #                                             1,
                            #                                             model.netSeg_A,
                            #                                             overlap=0.66,
                            #                                             mode="gaussian",
                            #                                             sigma_scale=[0.128, 0.128,0.01])



                        # seg = from_engine(["pred"])(vol_data)
                        #print("seg length: ", len(seg))
                        # seg = seg[0]
                        #print("seg shape: ", np.shape(seg))
                        # seg = np.array(seg)
                        # seg=np.squeeze(seg)
                        # seg[seg >= 0.5]=1.0
                        # seg[seg < 0.5]=0.0



                        # vol_data = [post_transforms(i) for i in decollate_batch(vol_data)]
                        # seg_out= from_engine(["pred"])(vol_data)[0]
                        # seg_out = np.array(seg_out)
                        # seg_out=np.squeeze(seg_out)
                        # seg_out[seg_out >= 0.5]=1.0
                        # seg_out[seg_out < 0.5]=0.0
                        # seg_out = np.transpose(seg_out, (2, 1, 0))


                        # cur_rd_path=os.path.join(valpath,img_name)
                        # im_obj = sitk.ReadImage(cur_rd_path)
                        # seg_out = sitk.GetImageFromArray(seg_out)
                        # seg_out = copy_info(im_obj, seg_out)
                        # sitk.WriteImage(seg_out, 'seg_%s' % img_name)


                    model.save('AVG_best_finetuned')
                    best_dice = val_avg_dice_2D
        if lr>0:
                model.update_learning_rate()