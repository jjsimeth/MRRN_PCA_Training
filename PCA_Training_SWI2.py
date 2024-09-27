# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:41:09 2023

@author: SimethJ
"""



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
from options.train_options import TrainingOptions
from models.models import create_model
# from util import util
from PCA_DIL_inference_utils import sliding_window_inference 


# from pathlib import Path
from glob import glob

from typing import Tuple
import monai
from monai.handlers.utils import from_engine
from monai.data import DataLoader, create_test_image_3d
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
)

opt = TrainingOptions().parse()
opt.isTrain=True

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


mr_paths = []
seg_paths = []

#Define input dimensions and resolution for inference model 
PD_in=np.array([0.6250, 0.6250, 3]) # millimeters
DIM_in=np.array([128,128,opt.nslices]) # 128x128 5 Slices
nmodalities=2

root_dir=os.getcwd()
opt.nchannels=opt.nslices*nmodalities

model = create_model(opt) 

model_path = os.path.join(root_dir,'deep_model','MRRNDS_model') #load weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_MR_seg_A(model_path) #use weights
# model.to(device)

#freeze all layers
for param in model.netSeg_A.parameters():
        param.requires_grad = False

                
#impath = os.path.join(path,'nii_vols') #load weights
impath = os.path.join(root_dir,'training_data') #load weights


#unfreeze final layer and Channel input block for finetuning
model.netSeg_A.CNN_block1.requires_grad_(True)
#model.netSeg_A.out_conv.requires_grad_(True)
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


#get nii and seg data
#nmodalities=2
# images = sorted(glob(os.path.join(root_dir, "adc*.nii.gz")))
images = sorted(glob(os.path.join(impath, "*_ep2d_diff_*.nii.gz"))) #adc keywords from filename
#segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
segs = sorted(glob(os.path.join(impath, "*t2_tse_tra*_ROI.nii.gz")))
# images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
images_t2w = sorted(glob(os.path.join(impath, "*_t2_tse*.nii.gz"))) #t2w keywords from filename

# print(images)
# print(segs)
# print(images_t2w)
#can add additional modalities with thier keyname
#images_ktrans = sorted(glob(os.path.join(root_dir, "ktrans*.nii.gz")))

n_val=3
n_train=7
train_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(images[:n_train], segs[:n_train], images_t2w[:n_train])] #first n_train to training
val_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(images[-n_val:], segs[-n_val:], images_t2w[-n_val:])] #last  n_val to validation

# print(train_files)
# print(val_files)

train_transforms = Compose(
    [
        LoadImaged(keys=["img","t2w","seg"]),
        EnsureChannelFirstd(keys=["img","t2w", "seg"]),
        Orientationd(keys=["img","t2w","seg"], axcodes="RAS"),     
        
        ResampleToMatchd(keys=["img"],
                    key_dst="t2w",
                    mode="bilinear"),
        # ResampleToMatchd(keys=["seg"],
        #             key_dst="t2w",
        #             mode="nearest"),
        Spacingd(keys=["img","t2w"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="bilinear"),
        Spacingd(keys=["seg"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="nearest"),
        
        ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
        #RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
        #CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
        #ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
        #RandRotate90d(keys=["img","t2w","seg"], prob=0.5, spatial_axes=[0, 1]),
        #CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),
        CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[96,96,2]),
        RandRotated(keys=["img","t2w","seg"], prob=0.9, range_x=3.0),
        RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),
        Transposed(keys=["img", "seg","t2w"], indices =[3,2,1,0]),
        SqueezeDimd(keys=["img","t2w", "seg"],dim=-1), 
        EnsureTyped(keys=["img","t2w", "seg"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["img","t2w","seg"]),
        EnsureChannelFirstd(keys=["img","t2w", "seg"]),
        Orientationd(keys=["img","t2w","seg"], axcodes="RAS"),     
        
        ResampleToMatchd(keys=["img"],
                    key_dst="t2w",
                    mode="bilinear"),
        # ResampleToMatchd(keys=["seg"],
        #             key_dst="t2w",
        #             mode="nearest"),
        Spacingd(keys=["img","t2w"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="bilinear"),
        Spacingd(keys=["seg"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="nearest"),
        
        ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
        #RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
        #CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
        #ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
        #RandRotate90d(keys=["img","t2w","seg"], prob=0.5, spatial_axes=[0, 1]),
        #RandRotated(keys=["img","t2w","seg"], prob=0.9, range_x=3.0),
        #CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),
        CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[96,96,2]),
        #RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),
        EnsureTyped(keys=["img","t2w", "seg"]),
    ]
)

post_transforms = Compose([
        EnsureTyped(keys="pred"),
        # Invertd(
        #     keys="pred",
        #     transform=val_transforms,
        #     orig_keys="t2w",
        #     meta_keys="pred_meta_dict",
        #     orig_meta_keys="image_meta_dict",
        #     meta_key_postfix="meta_dict",
        #     nearest_interp=True,
        #     to_tensor=True,
        # ),
        AsDiscreted(keys="pred", argmax=False),

    ])


# 3D dataset with preprocessing transforms
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)


train_loader = DataLoader(
    train_ds,
    batch_size=5,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)


check_data = monai.utils.misc.first(train_loader)
print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

epoch_loss_values = []
num_epochs = 10000
best_dice=0
for epoch in range(num_epochs):
    if epoch==100:
        model.netSeg_A.out_conv.requires_grad_(True)
    if epoch==500:    
        model.netSeg_A.CNN_block2.requires_grad_(True)
        model.netSeg_A.RU11.requires_grad_(True)
        
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{num_epochs}")
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        adc, labels, t2w = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["t2w"].to(device)
        
        if torch.sum(labels)>0:
            adc=scale_ADC(adc)
            
            inputs=torch.cat((adc,t2w),dim=1)
            labels=labels[:,2,:,:]
            
    
            
            inputs, labels = mixup(inputs, labels, np.random.beta(1.0, 1.0))
            #inputs, labels = mixup(inputs, labels, np.random.beta(0.2, 0.2))
            labels=  torch.clamp(labels,0.001,0.999) #label smoothing
            
            model.set_input_sep(inputs,labels)
            model.optimize_parameters()
    if (epoch%50)==0:
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        with torch.no_grad(): # no grade calculation 
            dice_2D=[]
            smooth=1
            step=0
            for val_data in val_loader:
                step += 1
                adc, label_val,t2w = val_data["img"].to(device), val_data["seg"].to(device), val_data["t2w"].to(device)
                
                label_val_vol=label_val
                #if torch.sum(label_val)>0:
                adc=scale_ADC(adc)
                
                val_inputs=torch.cat((adc,t2w),dim=1)
               
                
                with autocast(enabled=True):
                    #pass model segmentor and region info
                    #input, roi size, sw batch size, model, overlap (0.5)
                    val_data["pred"] = sliding_window_inference(val_inputs,
                                                                (128, 128, 5),
                                                                1,
                                                                model.netSeg_A,
                                                                overlap=0.50,
                                                                mode="gaussian",
                                                                sigma_scale=[0.128, 0.128,0.001])
            
                seg = from_engine(["pred"])(val_data)
                #print("seg length: ", len(seg))
                seg = seg[0]
                #print("seg shape: ", np.shape(seg))
                seg = np.array(seg)
                seg=np.squeeze(seg)
                seg[seg >= 0.5]=1.0
                seg[seg < 0.5]=0.0
                
                
                
                
               
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
                
    
                
                
                seg_temp=np.array(seg)
                gt_temp=np.array(label_val_vol.cpu())
    
                seg_flt=seg_temp.flatten()
                gt_flt=gt_temp.flatten()
                
                
                intersection = np.sum(seg_flt * (gt_flt > 0))
                dice_2D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
                print(dice_2D_temp)
                dice_2D.append(dice_2D_temp)
                
                
                
                
                
            dice_2D=np.average(dice_2D)
            print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D, ' (best: %.2f)'  % best_dice)
            if dice_2D>best_dice:
                    print ('saving for Dice, %.2f' % dice_2D, ' > %.2f' % best_dice) 
                    im_iter=0
                    for vol_data in val_loader:
                        im_iter += 1
                        adc, label_val,t2w = vol_data["img"].to(device), vol_data["seg"].to(device), vol_data["t2w"].to(device)
                        
                        label_val_vol=label_val
                        #if torch.sum(label_val)>0:
                        adc=scale_ADC(adc)
                        
                        val_inputs=torch.cat((adc,t2w),dim=1)
                       
                        
                        with autocast(enabled=True):
                            #pass model segmentor and region info
                            #input, roi size, sw batch size, model, overlap (0.5)
                            vol_data["pred"] = sliding_window_inference(val_inputs,
                                                                        (128, 128, 5),
                                                                        1,
                                                                        model.netSeg_A,
                                                                        overlap=0.50,
                                                                        mode="gaussian",
                                                                        sigma_scale=[0.128, 0.128,0.001])
                    
                        seg = from_engine(["pred"])(vol_data)
                        #print("seg length: ", len(seg))
                        seg = seg[0]
                        #print("seg shape: ", np.shape(seg))
                        seg = np.array(seg)
                        seg=np.squeeze(seg)
                        seg[seg >= 0.5]=1.0
                        seg[seg < 0.5]=0.0
                    
                        sitk.WriteImage(sitk.GetImageFromArray(t2w.cpu()), 'val%i_t2w.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(adc.cpu()), 'val%i_adc.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(label_val.cpu()), 'val%i_gtv.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(seg), 'val%i_seg.nii.gz' % im_iter)

                    model.save('AVG_best_finetuned')
                    best_dice = dice_2D