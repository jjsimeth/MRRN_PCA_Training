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
from options.train_options import TrainOptions
from models.models import create_model
# from util import util
from PCA_DIL_inference_utils import sliding_window_inference 

opt = TrainOptions().parse()
opt.isTrain=True

# from pathlib import Path
from glob import glob

from typing import Tuple
import monai
from monai.handlers.utils import from_engine
from monai.data import DataLoader, create_test_image_3d,PatchDataset
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
    RandSpatialCropSamplesd,
    RandSimulateLowResolutiond,
    OneOf,
    RandCropByPosNegLabeld,
)
import re
import json
from collections import defaultdict

def find_common_files(folder_paths, custom_names, pattern=r'P(\d+)_S(\d+)_.*\.nii\.gz'):
    if len(folder_paths) != len(custom_names):
        raise ValueError("The number of folder paths must match the number of custom names.")
    
    file_dict = defaultdict(lambda: defaultdict(list))
    
    for folder, name in zip(folder_paths, custom_names):
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' does not exist.")
            continue
        
        for file in os.listdir(folder):
            match = re.match(pattern, file)

            if match:
                id1 = match.group(1)
                id2 = match.group(2) if match.lastindex and match.lastindex >= 2 else 99
                file_id = f'P{id1}_S{id2}'
                file_dict[file_id][name].append(os.path.join(folder, file))
    
    # Find IDs that exist in all folders
    common_ids = [fid for fid, paths in file_dict.items() if len(paths) == len(folder_paths)]
    
    result_files = {fid: {name: file_dict[fid][name][0] for name in custom_names} for fid in common_ids}
    
    val_files = [{name: result_files[file_id][name] for name in custom_names} for file_id in common_ids]
    
    return val_files


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



mr_paths = []
seg_paths = []

#Define input dimensions and resolution for inference model 
PD_in=np.array([0.6250, 0.6250, 3]) # millimeters
DIM_in=np.array([128,128,opt.nslices]) # 128x128 5 Slices
nmodalities=1

root_dir=os.getcwd()
opt.nchannels=opt.nslices*nmodalities

model = create_model(opt) 

model_path = os.path.join(root_dir,'deep_model','MRRNDS_model') #load weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.load_MR_seg_A(model_path) #use weights
# model.to(device)

#freeze all layers
#for param in model.netSeg_A.parameters():
#        param.requires_grad = False

                
#impath = os.path.join(path,'nii_vols') #load weights
# impath = os.path.join(root_dir,'training_data') #load weights
# valpath = os.path.join(root_dir,'validation_data') #load weights


#unfreeze final layer and Channel input block for finetuning
# model.netSeg_A.CNN_block1.requires_grad_(True)
# model.netSeg_A.out_conv.requires_grad_(True)
# model.netSeg_A.out_conv1.requires_grad_(True)
# if opt.deeplayer>0:
#     model.netSeg_A.deepsupconv.requires_grad_(True)
dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(root_dir,opt.name,'ct_seg_val_loss.csv')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
fd_results = open(wt_path, 'w')
fd_results.write('train loss, seg accuracy,\n')

results_path=os.path.join(root_dir,opt.name,'val_results')
if not os.path.exists(results_path):
    os.makedirs(results_path)   
seg_path=os.path.join(root_dir,opt.name,'Output_segs')
if not os.path.exists(seg_path):
    os.makedirs(seg_path)    

datadir= r'/lila/data/deasy/Josiah_data/Prostate/nii_data'   

adcpath = os.path.join(datadir,'MR_LINAC50_NII','Images','ADC')
t2path = os.path.join(datadir,'MR_LINAC50_NII','Images','T2w')
trainpath_masks = os.path.join(datadir,'MR_LINAC50_NII','Masks','DIL')
trainpath_prostate_masks = os.path.join(datadir,'MR_LINAC50_NII','Masks','Prostate')


adcpath2 = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Images','ADC')
t2path2 = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Images','T2w')
trainpath2_masks = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Masks','DIL')
trainpath3_masks = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Masks','DIL_T2w')
trainpath2_prostate_masks = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Masks','Prostate_JJS')

valpath_adcpath = os.path.join(datadir,'MR_LINAC10_NII','Images','ADC')
valpath_t2path = os.path.join(datadir,'MR_LINAC10_NII','Images','T2w')
valpath_masks = os.path.join(datadir,'MR_LINAC10_NII','Masks','GTV')
valpath_prostate_masks = os.path.join(datadir,'MR_LINAC10_NII','Masks','Prostate')




trainfolders = [adcpath, trainpath_masks,t2path,trainpath_prostate_masks]  # Replace with actual paths
trainfolders2 = [adcpath2, trainpath2_masks,t2path2,trainpath2_prostate_masks]  # Replace with actual paths
trainfolders3 =  [adcpath2, trainpath3_masks,t2path2,trainpath2_prostate_masks]  # Replace with actual paths

valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
custom_names = ["img", "seg", "t2w","prost"]
train_files = find_common_files(trainfolders, custom_names,pattern=r'.*P(\d+)_S(\d+).*\.nii\.gz')
train_files2 = find_common_files(trainfolders2, custom_names,pattern=r'.*P(\d+)_.*\.nii\.gz')
train_files3 = find_common_files(trainfolders3, custom_names,pattern=r'.*P(\d+)_.*\.nii\.gz')

val_files = find_common_files(valfolders, custom_names,pattern=r'.*P(\d+)_S(\d+).*\.nii\.gz')


train_files=train_files+train_files2+train_files3

print("Matching files:")
for file_entry in train_files:
    print(file_entry)


# print(train_files)
# print(val_files)

train_transforms = Compose(
    [
            LoadImaged(keys=["img", "t2w", "prost","seg"]),
            EnsureChannelFirstd(keys=["img", "t2w","prost", "seg"]),
            Orientationd(keys=["img", "t2w","prost", "seg"], axcodes="RAS"),

            # ResampleToMatchd(keys=["img"],
            #                  key_dst="t2w",
            #                  mode="bilinear"),
            # ResampleToMatchd(keys=["seg"],
            #                  key_dst="t2w",
            #                  mode="nearest"),
            ResampleToMatchd(keys=["img","prost","seg"],
                             key_dst="t2w",
                             mode=("bilinear","nearest", "nearest")),
            Spacingd(keys=["img", "t2w", "prost","seg"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode=("bilinear", "bilinear","nearest", "nearest")),
            #CenterSpatialCropd(keys=["img", "t2w", "seg"], roi_size=[200, 200, -1]),
            
            CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[256, 256,-1]),
            
            CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="seg", margin=[256, 256, opt.extra_neg_slices + (opt.nslices - 1) / 2],allow_smaller=True),
            CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="prost", margin=[64, 64, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            
            
            #CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="prost", margin=[64, 64, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            #CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="prost", margin=[64, 64, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            
            
            # ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
            OneOf(transforms=[
                RandAdjustContrastd(keys=["img"], prob=0.25),
                RandHistogramShiftd(keys=["img"], prob=0.25),
                RandBiasFieldd(keys=["img"], prob=0.25),
                RandSimulateLowResolutiond(keys=["img"], prob=0.25),
                RandGaussianNoised(keys=["img"], prob=0.25),
                RandGaussianSmoothd(keys=["img"], prob=0.25),
                ]),
                
            OneOf(transforms=[
                RandAdjustContrastd(keys=["t2w"], prob=0.25),
                RandHistogramShiftd(keys=["t2w"], prob=0.25),
                RandBiasFieldd(keys=["t2w"], prob=0.25),
                RandSimulateLowResolutiond(keys=["t2w"], prob=0.25),
                RandGaussianNoised(keys=["t2w"], prob=0.25),
                RandGaussianSmoothd(keys=["t2w"], prob=0.25),
                ]),
            
  

            ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True,channel_wise=True),
            ScaleIntensityRangePercentilesd(keys=["t2w"], lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True,channel_wise=True),
            #RandCropByPosNegLabeld(keys=["img", "t2w", "prost", "seg"], label_key="seg", spatial_size=(128, 128,opt.nslices)),
            
            
            RandRotated(keys=["img", "t2w", "prost", "seg"], prob=1.0, range_x=1.0, range_y=0.0, range_z=0.0, mode=("bilinear", "bilinear","nearest", "nearest")),
            CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[142, 142,-1]),
            #CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="seg", margin=[96, 96, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            
            
            #RandSpatialCropSamplesd(keys=["img", "t2w", "prost", "seg"], num_samples=1, roi_size=(128, 128,opt.nslices),random_size=False,random_center=True),
            RandSpatialCropd(keys=["img", "t2w", "prost",  "seg"], roi_size=(128, 128, opt.nslices), random_size=False,random_center =True),
            
            RandFlipd(keys=["img", "t2w", "prost", "seg"], prob=0.5),
            RandAxisFlipd(keys=["img", "t2w", "prost", "seg"], prob=0.25),
            
           
            Transposed(keys=["img", "t2w",  "prost", "seg"], indices=[3, 2, 1, 0]),
            SqueezeDimd(keys=["img", "t2w",  "prost", "seg"], dim=-1),
            EnsureTyped(keys=["img", "t2w",  "prost", "seg"]),

    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["img", "t2w", "prost","seg"]),
        EnsureChannelFirstd(keys=["img", "t2w","prost", "seg"]),
        Orientationd(keys=["img", "t2w","prost", "seg"], axcodes="RAS"),

        # ResampleToMatchd(keys=["img"],
        #                  key_dst="t2w",
        #                  mode="bilinear"),
        # ResampleToMatchd(keys=["seg"],
        #                  key_dst="t2w",
        #                  mode="nearest"),
        ResampleToMatchd(keys=["t2w","prost","seg"],
                         key_dst="img",
                         mode=("bilinear","nearest", "nearest")),
        Spacingd(keys=["img", "t2w", "prost","seg"],
                 pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                 mode=("bilinear", "bilinear","nearest", "nearest")),
        #CenterSpatialCropd(keys=["img", "t2w", "seg"], roi_size=[200, 200, -1]),
        
        CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[256, 256,-1]),
        CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="seg", margin=[256, 256, opt.extra_neg_slices + (opt.nslices - 1) / 2],allow_smaller=True),
        CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="prost", margin=[64, 64, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
        
        #CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="prost", margin=[64, 64, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
        CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[142, 142,-1]),
            
        ScaleIntensityRangePercentilesd(keys=["img"], lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True,channel_wise=True),
        ScaleIntensityRangePercentilesd(keys=["t2w"], lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True,channel_wise=True),
        EnsureTyped(keys=["img","t2w","prost",  "seg"]),
    ]
)


post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="img",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=False),

    ])






train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

train_ds = monai.data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_num= 200,#400,#1200,#400,#200,#args.cache_num,# 250,
            cache_rate=1.0,
            num_workers=4,
        )
val_ds = monai.data.CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num= 100,#400,#1200,#400,#200,#args.cache_num,# 250,
            cache_rate=1.0,
            num_workers=4,
        )



train_loader = DataLoader(
    train_ds,
    batch_size=opt.batchSize,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    drop_last=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    drop_last=True
)


check_data = monai.utils.misc.first(train_loader)
print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

sitk.WriteImage(sitk.GetImageFromArray(check_data["t2w"].cpu()[0,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_t2w.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["img"].cpu()[0,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_adc.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["seg"].cpu()[0,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_SEG.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["prost"].cpu()[0,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_prost.nii.gz'))


num_epochs=opt.niter+opt.niter_decay
epoch_loss_values = []
best_dice=0
mixup_ab=opt.mixup_betadist
for epoch in range(num_epochs+1):
    lr=model.get_curr_lr()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1

        adc, labels, t2w, prost = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["t2w"].to(device), batch_data["prost"].to(device)

        #inputs=torch.cat((adc,t2w),dim=1)
        inputs=t2w
        labels=labels[:,np.int32((opt.nslices-1)/2),:,:]
        
        labels=  torch.clamp(labels,0.001,0.999).float()
        inputs, labels = mixup(inputs, labels, np.random.beta(mixup_ab, mixup_ab))
        model.set_input_sep(inputs,labels)
        model.optimize_parameters()
        
        
    if (epoch%opt.display_freq)==0:      
                        errors = model.get_current_errors()['Seg_loss']
                        print (errors)   
        
        
    if (epoch%opt.display_freq)==0:
        print("-" * 10)
        print(f"epoch {epoch}/{num_epochs}")
        with torch.no_grad(): # no grade calculation 
            dice_2D=[]
            smooth=1
            step=0
            for val_data in val_loader:
                step += 1
                adc, label_val,t2w,prost = val_data["img"].to(device), val_data["seg"].to(device), val_data["t2w"].to(device), val_data["prost"].to(device)
                label_val_vol=label_val
                #val_inputs=torch.cat((adc,t2w),dim=1)
                val_inputs=t2w
               
                
                with autocast(enabled=True):
                    #pass model segmentor and region info
                    #input, roi size, sw batch size, model, overlap (0.5)
                    val_data["pred"] = sliding_window_inference(val_inputs,
                                                                (128, 128, opt.nslices),
                                                                4,
                                                                model.netSeg_A,
                                                                overlap=1-(1/opt.nslices),
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
                
                seg_temp=np.array(seg)
                gt_temp=np.array(label_val_vol.cpu())
    
                seg_flt=seg_temp.flatten()
                gt_flt=gt_temp.flatten()
                
                
                intersection = np.sum(seg_flt * (gt_flt > 0))
                dice_2D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
                print('  DSC:  %.2f' % dice_2D_temp)
                dice_2D.append(dice_2D_temp)
                
                
                
                
                
            dice_2D=np.average(dice_2D)
            print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D, ' (best: %.2f)'  % best_dice)
            
            fd_results.write(str(dice_2D) + '\n')
            fd_results.flush()  
            
                    
            if dice_2D>best_dice:
                    print ('saving for Dice, %.2f' % dice_2D, ' > %.2f' % best_dice) 
                    model.save('AVG_best_finetuned')
                    best_dice = dice_2D
                    im_iter=0
                    for vol_data in val_loader:
                        im_iter += 1
                        adc, label_val,t2w,prost = vol_data["img"].to(device), vol_data["seg"].to(device), vol_data["t2w"].to(device), vol_data["prost"].to(device)
                        
                        img_name=adc.meta['filename_or_obj'][0].split('/')[-1]
                        print(img_name)
                        
                        label_val_vol=label_val
                        #if torch.sum(label_val)>0:
                        # adc=scale_ADC(adc)
                        
                        #val_inputs=torch.cat((adc,t2w),dim=1)
                        val_inputs=t2w
                        
                        with autocast(enabled=True):
                            #pass model segmentor and region info
                            #input, roi size, sw batch size, model, overlap (0.5)
                            vol_data["pred"] = sliding_window_inference(val_inputs,
                                                                        (128, 128, opt.nslices),
                                                                        1,
                                                                        model.netSeg_A,
                                                                        overlap=0.66,
                                                                        mode="gaussian",
                                                                        sigma_scale=[0.128, 0.128,0.01])
                        
                        
                        
                        seg = from_engine(["pred"])(vol_data)
                        #print("seg length: ", len(seg))
                        seg = seg[0]
                        #print("seg shape: ", np.shape(seg))
                        seg = np.array(seg)
                        seg=np.squeeze(seg)
                        seg[seg >= 0.5]=1.0
                        seg[seg < 0.5]=0.0
                        
                        sitk.WriteImage(sitk.GetImageFromArray(prost.cpu()), os.path.join(results_path,'val%i_prost.nii.gz' % im_iter))
                        sitk.WriteImage(sitk.GetImageFromArray(t2w.cpu()), os.path.join(results_path,'val%i_t2w.nii.gz' % im_iter))
                        sitk.WriteImage(sitk.GetImageFromArray(adc.cpu()),  os.path.join(results_path,'val%i_adc.nii.gz' % im_iter))
                        sitk.WriteImage(sitk.GetImageFromArray(label_val.cpu()),  os.path.join(results_path,'val%i_gtv.nii.gz' % im_iter))
                        sitk.WriteImage(sitk.GetImageFromArray(seg),  os.path.join(results_path,'val%i_seg.nii.gz' % im_iter))
                        
                        vol_data = [post_transforms(i) for i in decollate_batch(vol_data)]
                        seg_out= from_engine(["pred"])(vol_data)[0]
                        seg_out = np.array(seg_out)
                        seg_out=np.squeeze(seg_out)
                        seg_out[seg_out >= 0.5]=1.0
                        seg_out[seg_out < 0.5]=0.0
                        seg_out = np.transpose(seg_out, (2, 1, 0))
                        
                        
                        cur_rd_path=os.path.join(valpath_adcpath,img_name)
                        im_obj = sitk.ReadImage(cur_rd_path)
                        seg_out = sitk.GetImageFromArray(seg_out)
                        seg_out = copy_info(im_obj, seg_out)
                        sitk.WriteImage(seg_out,  os.path.join(seg_path,'seg_%s' % img_name))



    if lr>0:
            model.update_learning_rate()