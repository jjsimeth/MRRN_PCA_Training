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
import numpy as np
#import matplotlib.pyplot as plt
import os

import torch.multiprocessing

import torch.utils.data
from options.train_options import TrainingOptions
from models.models import create_model
from util import util


from pathlib import Path
from glob import glob

import monai
from monai.data import DataLoader, create_test_image_3d
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
    Transposed
)

opt = TrainingOptions().parse()
opt.isTrain=True




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
model.netSeg_A.out_conv.requires_grad_(True)

#we can unfreeze other layers either right away or on a epoch based schedule

#we likely want to train a little bit just to get the channel adaptation working 
#then if we have a lot of data unfreeze more layers as below:
   
#first 5 layers
    # 
    # model.netSeg_A.CNN_block2.requires_grad_(True)
    # model.netSeg_A.CNN_block3.requires_grad_(True)
    # model.netSeg_A.RU1.requires_grad_(True)
    # model.netSeg_A.RU2.requires_grad_(True)
    # model.netSeg_A.RU3.requires_grad_(True)

#last 3 layers (not counting output conv)    
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
segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
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


train_transforms = Compose(
    [
        LoadImaged(keys=["img","t2w","seg"]),
        EnsureChannelFirstd(keys=["img","t2w", "seg"]),
        Orientationd(keys=["img","t2w","seg"], axcodes="RAS"),     
        
        Spacingd(keys=["img","t2w"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="bilinear"),
        Spacingd(keys=["seg"],
                    pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                    mode="nearest"),
        ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
        #ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
        RandRotate90d(keys=["img","t2w","seg"], prob=0.5, spatial_axes=[0, 1]),
        CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),
        EnsureTyped(keys=["img","t2w", "seg"]),

    ]
)

# 3D dataset with preprocessing transforms
volume_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
volume_validation = monai.data.CacheDataset(data=val_files, transform=train_transforms)
# use batch_size=1 to check the volumes because the input volumes have different shapes
check_loader = DataLoader(volume_ds, batch_size=1)
check_data = monai.utils.misc.first(check_loader)
print("first volume's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)


patch_func = Compose(
    [
     monai.data.PatchIterd(keys=["img","seg","t2w"], patch_size=( None, None, opt.nslices), start_pos=(0, 0, 0)), #using 5 slices like I did, but you can change to 1 or 3 if desired
     #monai.data.PatchIterd(keys=[], patch_size=(1, None, None), start_pos=(0, 0, 0)) #using 1 target slice, but you could have it match image slices if desired (just need to use overlapping sliding window inference)
         ]
)
patch_transform = Compose(
    [
         # squeeze the last dim
        #Resized(keys=["img","t2w", "seg"], spatial_size=[-1, DIM_in[0], DIM_in[1]]),
        # to use crop/pad instead of resize:
        ResizeWithPadOrCropd(keys=["img", "seg","t2w"], spatial_size=[DIM_in[0], DIM_in[1], None], mode="replicate"),
        Transposed(keys=["img", "seg","t2w"], indices =[3,2,1,0]),
        SqueezeDimd(keys=["img","t2w", "seg"],dim=-1), 
    ]
)
patch_ds = monai.data.GridPatchDataset(
    data=volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False
)

patch_val = monai.data.GridPatchDataset(
    data=volume_validation, patch_iter=patch_func, transform=patch_transform, with_coordinates=False
)

shuffle_ds = monai.data.ShuffleBuffer(patch_ds, buffer_size=30, seed=0)
train_loader = DataLoader(
    shuffle_ds,
    batch_size=3,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    patch_val,
    batch_size=1,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)


check_data = monai.utils.misc.first(train_loader)
print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

epoch_loss_values = []
num_epochs = 5
best_dice=0
for epoch in range(num_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        adc, labels,t2w = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["t2w"].to(device)
        adc=scale_ADC(adc)
        
        inputs=torch.cat((adc,t2w),dim=1)
        labels=labels[:,2,:,:]
            
        model.set_input_sep(inputs,labels)
        model.optimize_parameters()

    with torch.no_grad(): # no grade calculation 
        dice_2D=[]
        smooth=1
        for batch_data in val_loader:
            step += 1
            adc, label_val,t2w = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["t2w"].to(device)
            adc=scale_ADC(adc)
            
            inputs=torch.cat((adc,t2w),dim=1)
            label_val=label_val[:,2,:,:]
            
            model.set_test_input(inputs)
            ori_img, seg = model.net_Segtest_image()
            
            seg = np.array(seg)
            seg=np.squeeze(seg)
            seg = (seg > 0.5)
            
            seg_temp=np.array(seg)
            gt_temp=np.array(label_val.cpu())

            seg_flt=seg_temp.flatten()
            gt_flt=gt_temp.flatten()
            
            
            intersection = np.sum(seg_flt * (gt_flt > 0))
            dice_2D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
            dice_2D.append(dice_2D_temp)
        dice_2D=np.average(dice_2D)
        print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D)
        if dice_2D>best_dice:
                print ('saving for Dice, %.2f' % dice_2D, ' > %.2f' % best_dice)         
                model.save('AVG_best_finetuned')
                best_dice = dice_2D