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
#model.load_MR_seg_A(model_path) #use weights
# model.to(device)

#freeze all layers
#for param in model.netSeg_A.parameters():
#        param.requires_grad = False

                
#impath = os.path.join(path,'nii_vols') #load weights
impath = os.path.join(root_dir,'training_data') #load weights
valpath = os.path.join(root_dir,'validation_data') #load weights


#unfreeze final layer and Channel input block for finetuning
# model.netSeg_A.CNN_block1.requires_grad_(True)
# model.netSeg_A.out_conv.requires_grad_(True)
# model.netSeg_A.out_conv1.requires_grad_(True)
# if opt.deeplayer>0:
#     model.netSeg_A.deepsupconv.requires_grad_(True)
    
    
wt_path=root_dir+'ct_seg_val_loss.csv'

fd_results = open(wt_path, 'w')
fd_results.write('train loss, seg accuracy,\n')

    
    
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


# #get nii and seg data
# #nmodalities=2
# # images = sorted(glob(os.path.join(root_dir, "adc*.nii.gz")))
# images = sorted(glob(os.path.join(impath, "*[_ep2d_diff_*.nii.gz][*ADC_p*.nii]"))) #adc keywords from filename
# #segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
# segs = sorted(glob(os.path.join(impath, "*[t2_tse_tra*_ROI.nii.gz][T2w_p*.nii]")))
# # images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
# images_t2w = sorted(glob(os.path.join(impath, "*[_t2_tse*.nii.gz][GTV_p*.nii]"))) #t2w keywords from filename



types = ('ProstateX*_ep2d_diff_*.nii.gz', 'MSK_MR_*_ADC.nii.gz') # the tuple of file types
images=[]
for fname in types:
   images.extend(glob(os.path.join(impath, fname)))
images = sorted(images)

types = ('ProstateX*Finding*t2_tse_tra*_ROI.nii.gz', 'MSK_MR_*_GTV.nii.gz') # the tuple of file types
segs=[]
for fname in types:
   segs.extend(glob(os.path.join(impath, fname)))
segs = sorted(segs)
   
types = ('ProstateX*_t2_tse*.nii.gz', 'MSK_MR_*T2w.nii.gz') # the tuple of file types
images_t2w=[]
for fname in types:
   images_t2w.extend(glob(os.path.join(impath, fname)))
images_t2w = sorted(images_t2w)

# #val_images = (glob(os.path.join(valpath, "*_ep2d_diff_*.nii.gz"))+glob(os.path.join(impath, "*ADC_p*.nii"))) #adc keywords from filename

# #val_images = (glob(os.path.join(valpath, "*_ep2d_diff_*.nii.gz"))+glob(os.path.join(impath, "*ADC_p*.nii"))) #adc keywords from filename
# val_images = glob(os.path.join(valpath, "*_ep2d_diff_*.nii.gz")) #adc keywords from filename
# val_images.extend(glob(os.path.join(impath, "*ADC_p*.nii")))
# val_images = sorted(val_images)
# #segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
# #val_segs = sorted(glob(os.path.join(valpath, "*t2_tse_tra*_ROI.nii.gz"))+glob(os.path.join(impath, "*T2w_p*.nii")))

# val_segs = glob(os.path.join(valpath, "*t2_tse_tra*_ROI.nii.gz")) #adc keywords from filename
# val_segs.extend(glob(os.path.join(impath, "*T2w_p*.nii")))
# val_segs = sorted(val_segs)


# # images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
# #val_images_t2w = sorted(glob(os.path.join(valpath, "*_t2_tse*.nii.gz"))+glob(os.path.join(impath, "*GTV_p*.nii"))) #t2w keywords from filename
# val_images_t2w = glob(os.path.join(valpath, "*_t2_tse*.nii.gz")) #adc keywords from filename
# val_images_t2w.extend(glob(os.path.join(impath, "*GTV_p*.nii")))
# val_images_t2w = sorted(val_images_t2w)




types = ('ProstateX*_ep2d_diff_*.nii.gz', 'MSK_MR_*_ADC.nii.gz') # the tuple of file types
val_images=[]
for fname in types:
   val_images.extend(glob(os.path.join(valpath, fname)))
val_images = sorted(val_images)

types = ('ProstateX*Finding*t2_tse_tra*_ROI.nii.gz', 'MSK_MR_*_GTV.nii.gz') # the tuple of file types
val_segs=[]
for fname in types:
   val_segs.extend(glob(os.path.join(valpath, fname)))
val_segs = sorted(val_segs)
 

types = ('ProstateX*_t2_tse*.nii.gz', 'MSK_MR_*T2w.nii.gz') # the tuple of file types
val_images_t2w=[]
for fname in types:
   val_images_t2w.extend(glob(os.path.join(valpath, fname)))
val_images_t2w = sorted(val_images_t2w)





# print(images)
# print(segs)
# print(images_t2w)

print(val_images)
print(val_segs)
print(val_images_t2w)



#can add additional modalities with thier keyname
#images_ktrans = sorted(glob(os.path.join(root_dir, "ktrans*.nii.gz")))


# n_val=3
# n_train=7
train_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(images, segs, images_t2w)] #first n_train to training
val_files = [{"img": img, "seg": seg, "t2w": t2w} for img, seg, t2w in zip(val_images, val_segs, val_images_t2w)] #last  n_val to validation




# print(train_files)
# print(val_files)

train_transforms = Compose(
    [
            LoadImaged(keys=["img", "t2w", "seg"]),
            EnsureChannelFirstd(keys=["img", "t2w", "seg"]),
            Orientationd(keys=["img", "t2w", "seg"], axcodes="RAS"),

            # ResampleToMatchd(keys=["img"],
            #                  key_dst="t2w",
            #                  mode="bilinear"),
            # ResampleToMatchd(keys=["seg"],
            #                  key_dst="t2w",
            #                  mode="nearest"),
            ResampleToMatchd(keys=["img","seg"],
                             key_dst="t2w",
                             mode=("bilinear", "nearest")),
            Spacingd(keys=["img", "t2w", "seg"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode=("bilinear", "bilinear", "nearest")),
            CenterSpatialCropd(keys=["img", "t2w", "seg"], roi_size=[240, 240, -1]),
            
            RandFlipd(keys=["img", "t2w", "seg"], prob=0.5),
            #RandAxisFlipd(keys=["img", "t2w", "seg"], prob=0.1,),

            # ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
            RandAdjustContrastd(keys=["img"], prob=0.1),
            RandHistogramShiftd(keys=["img"], prob=0.1),
            RandBiasFieldd(keys=["img"], prob=0.1),
            RandAdjustContrastd(keys=["t2w"], prob=0.1),
            RandHistogramShiftd(keys=["t2w"], prob=0.1),
            RandBiasFieldd(keys=["t2w"], prob=0.1),
            
            RandGaussianNoised(keys=["img", "t2w"], prob=0.1),
            RandGaussianSmoothd(keys=["img", "t2w"], prob=0.1),
            
            
            ScaleIntensityRangePercentilesd(keys=["img", "t2w"], lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True),
            # RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
            # CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
            # ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
            # RandRotate90d(keys=["img","t2w","seg"], prob=0.2, spatial_axes=[0, 1]),
            # RandRotate90d(keys=["img","t2w","seg"], prob=0.2, spatial_axes=[1, 2]),
            
            # CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),

            # CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[-1,-1,opt.extra_neg_slices+(opt.nslices-1)/2]),
            

           
            
            RandRotated(keys=["img", "t2w", "seg"], prob=1.0, range_x=0.0, range_y=1.0, range_z=0.0, mode=("bilinear", "bilinear", "nearest")),
            RandRotated(keys=["img", "t2w", "seg"], prob=0.1, range_x=0.1, range_y=0.0, range_z=0.0, mode=("bilinear", "bilinear", "nearest")),
            RandRotated(keys=["img", "t2w", "seg"], prob=0.1, range_x=0.0, range_y=0.0, range_z=0.1, mode=("bilinear", "bilinear", "nearest")),
            
            CropForegroundd(keys=["img", "t2w", "seg"], source_key="seg", margin=[96, 96, opt.extra_neg_slices + (opt.nslices - 1) / 2]),
            
            
           
            RandSpatialCropd(keys=["img", "t2w", "seg"], roi_size=(128, 128, opt.nslices), random_size=False,random_center =True),
            Transposed(keys=["img", "seg", "t2w"], indices=[3, 2, 1, 0]),
            SqueezeDimd(keys=["img", "t2w", "seg"], dim=-1),
            EnsureTyped(keys=["img", "t2w", "seg"]),

    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["img","t2w","seg"]),
        EnsureChannelFirstd(keys=["img","t2w", "seg"]),
        Orientationd(keys=["img","t2w","seg"], axcodes="RAS"),     
        
        ResampleToMatchd(keys=["img","seg"],
                             key_dst="t2w",
                             mode=("bilinear", "nearest")),
        Spacingd(keys=["img", "t2w", "seg"],
                     pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                     mode=("bilinear", "bilinear", "nearest")),
        
        ScaleIntensityRangePercentilesd(keys=["img","t2w"],lower=0,upper=99,b_min=-1.0,b_max=1.0,clip=True),
        #ScaleIntensityd(keys="t2w",minv=-1.0, maxv=1.0),
        #RandCropByPosNegLabeld(["img", 'lbl'], "lbl", spatial_size=(32, 32)),
        #CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[256, 256,-1]),
        #ScaleIntensityd(keys="img",minv=-1.0, maxv=1.0),
        #RandRotate90d(keys=["img","t2w","seg"], prob=0.5, spatial_axes=[0, 1]),
        #RandRotated(keys=["img","t2w","seg"], prob=0.9, range_x=3.0),
        #CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[128,128,2]),
        #CenterSpatialCropd(keys=["img","t2w","seg"], roi_size=[160, 160,-1]),
        CropForegroundd(keys=["img","t2w","seg"], source_key= "seg", margin=[96,96,opt.extra_neg_slices+(opt.nslices-1)/2]),
        #RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),
        EnsureTyped(keys=["img","t2w", "seg"]),
    ]
)


# patch_transform = Compose(
#     [   #RandSpatialCropSamplesd(keys=["img","t2w","seg"],num_samples=10, roi_size=(128, 128, opt.nslices),random_size=False),
#         RandAdjustContrastd(keys=["img"],prob=0.2),
#         RandHistogramShiftd(keys=["img"],prob=0.2),
#         RandBiasFieldd(keys=["img"], prob=0.2),
#         RandAdjustContrastd(keys=["t2w"],prob=0.2),
#         RandHistogramShiftd(keys=["t2w"],prob=0.2),
#         RandBiasFieldd(keys=["t2w"], prob=0.2),
#         RandGaussianNoised(keys=["img","t2w"], prob=0.1),
#         RandGaussianSmoothd(keys=["img","t2w"], prob=0.1),
        
        
#         #SqueezeDimd(keys=["img","t2w", "seg"], dim=-1),  # squeeze the last dim
#         #Resized(keys=["img","t2w", "seg"], spatial_size=[48, 48]),
#         # to use crop/pad instead of resize:
#         # ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=[48, 48], mode="replicate"),
#     ]
# )

# num_samples=5    
# patch_func = monai.transforms.RandSpatialCropSamplesd(
#     keys=["img","t2w", "seg"],
#     #roi_size=[128, 128, opt.extra_neg_slices+(opt.nslices-1)/2],  # dynamic spatial_size for the first two dimensions
#     roi_size=[128, 128],
#     num_samples=num_samples,
#     random_size=False,
# )
    
# volume_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
# patch_ds = PatchDataset(
#     volume_ds,
#     transform=patch_transform,
#     patch_func=patch_func,
#     samples_per_image=num_samples,
# )


post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="t2w",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=False),

    ])


# 3D dataset with preprocessing transforms
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
#train_ds = monai.data.ShuffleBuffer(patch_ds, seed=0) #patch based trainer



val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)


train_loader = DataLoader(
    train_ds,
    batch_size=opt.batchSize,
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

sitk.WriteImage(sitk.GetImageFromArray(check_data["t2w"].cpu()[0,:,:,:]), 'TRAINING_DEBUG_t2w.nii.gz')
sitk.WriteImage(sitk.GetImageFromArray(check_data["img"].cpu()[0,:,:,:]), 'TRAINING_DEBUG_adc.nii.gz')
sitk.WriteImage(sitk.GetImageFromArray(check_data["seg"].cpu()[0,:,:,:]), 'TRAINING_DEBUG_SEG.nii.gz')



num_epochs=opt.niter+opt.niter_decay
epoch_loss_values = []
best_dice=0
mixup_ab=opt.mixup_betadist
for epoch in range(num_epochs+1):
    lr=model.get_curr_lr()
    # if epoch==100:
    #     model.netSeg_A.out_conv.requires_grad_(True)
    # if epoch==round(num_epochs*opt.unfreeze_fraction1):    
    #     print('  >>unfreeze layer 2 and RU11')
    #     model.netSeg_A.CNN_block2.requires_grad_(True)
    #     model.netSeg_A.RU11.requires_grad_(True)
    # if epoch==round(num_epochs*opt.unfreeze_fraction2):
    #     print('  >>unfreeze layer 3, RU1,RU2,RU3, RU33 and RU22')
    #     # model.netSeg_A.CNN_block2.requires_grad_(True)
    #     model.netSeg_A.CNN_block3.requires_grad_(True)
    #     model.netSeg_A.RU1.requires_grad_(True)
    #     model.netSeg_A.RU2.requires_grad_(True)
    #     model.netSeg_A.RU3.requires_grad_(True)
        
    #     #last 3 layers (not counting output conv)    
    #     model.netSeg_A.RU33.requires_grad_(True)
    #     model.netSeg_A.RU22.requires_grad_(True)
    #     # model.netSeg_A.RU11.requires_grad_(True)
    # if epoch==round(num_epochs*0.9):
    #     for param in model.netSeg_A.parameters():
    #         param.requires_grad = True
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{num_epochs}")
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        
        # img_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
        # print(img_name)
        
        adc, labels, t2w = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["t2w"].to(device)
        # img_name=t2w.meta['filename_or_obj']
        # print(img_name)
        
        #if torch.sum(labels)>0:
        # for ibatch in range(0,opt.batchSize+1):
        #     adc[ibatch,:]=scale_ADC(adc[ibatch,:])
        
        inputs=torch.cat((adc,t2w),dim=1)
        labels=labels[:,np.int32((opt.nslices-1)/2),:,:]
        #labels=labels[:,2,:,:]
        
        
        labels=  torch.clamp(labels,0.001,0.999).float()
        inputs, labels = mixup(inputs, labels, np.random.beta(mixup_ab, mixup_ab))
        # inputs, labels = mixup(inputs, labels, np.random.beta(mixup_ab, mixup_ab))
        
        #inputs, labels = mixup(inputs, labels, np.random.beta(0.2, 0.2))
         #label smoothing
        
        model.set_input_sep(inputs,labels)
        model.optimize_parameters()
        
        
    if (epoch%opt.display_freq)==0:      
    #                     # save_result = total_steps % opt.update_html_freq == 0
    #                     # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                        errors = model.get_current_errors()['Seg_loss']
                        
                        # message = '(epoch: %d) ' %epoch
                        # for k, v in errors.items():
                        #     message += '%s: %.3f ' % (k, v)
                        #     print(message)

                        #t = (time.time() - iter_start_time) / opt.batchSize
                        print (errors)
                        #visualizer.print_current_errors(epoch, epoch_iter, errors, t)    
        
        
    if (epoch%opt.display_freq)==0:
        print("-" * 10)
        print(f"epoch {epoch}/{num_epochs}")
        with torch.no_grad(): # no grade calculation 
            dice_2D=[]
            smooth=1
            step=0
            for val_data in val_loader:
                step += 1
                adc, label_val,t2w = val_data["img"].to(device), val_data["seg"].to(device), val_data["t2w"].to(device)
                
        
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
            
            fd_results.write(str(dice_2D) + '\n')
            fd_results.flush()  
            
                    
            if dice_2D>best_dice:
                    print ('saving for Dice, %.2f' % dice_2D, ' > %.2f' % best_dice) 
                    im_iter=0
                    for vol_data in val_loader:
                        im_iter += 1
                        adc, label_val,t2w = vol_data["img"].to(device), vol_data["seg"].to(device), vol_data["t2w"].to(device)
                        
                        img_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
                        print(img_name)
                        
                        label_val_vol=label_val
                        #if torch.sum(label_val)>0:
                        # adc=scale_ADC(adc)
                        
                        val_inputs=torch.cat((adc,t2w),dim=1)
                       
                        
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
                    
                        sitk.WriteImage(sitk.GetImageFromArray(t2w.cpu()), 'val%i_t2w.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(adc.cpu()), 'val%i_adc.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(label_val.cpu()), 'val%i_gtv.nii.gz' % im_iter)
                        sitk.WriteImage(sitk.GetImageFromArray(seg), 'val%i_seg.nii.gz' % im_iter)
                        
                        vol_data = [post_transforms(i) for i in decollate_batch(vol_data)]
                        seg_out= from_engine(["pred"])(vol_data)[0]
                        seg_out = np.array(seg_out)
                        seg_out=np.squeeze(seg_out)
                        seg_out[seg_out >= 0.5]=1.0
                        seg_out[seg_out < 0.5]=0.0
                        seg_out = np.transpose(seg_out, (2, 1, 0))
                        
                        
                        cur_rd_path=os.path.join(valpath,img_name)
                        im_obj = sitk.ReadImage(cur_rd_path)
                        seg_out = sitk.GetImageFromArray(seg_out)
                        seg_out = copy_info(im_obj, seg_out)
                        sitk.WriteImage(seg_out, 'seg_%s' % img_name)


                    model.save('AVG_best_finetuned')
                    best_dice = dice_2D
    if lr>0:
            model.update_learning_rate()