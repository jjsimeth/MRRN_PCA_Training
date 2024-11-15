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
from scipy import ndimage
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
opt.isTrain=False

# from pathlib import Path
from glob import glob
from util import HD
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
)

from scipy.ndimage.measurements import label

def lesion_eval(seg,gtv,spacing_mm):
  #filter out small unconnected segs
    labeled_gtv=np.squeeze(gtv)
    labeled_gtv[labeled_gtv>0.0]=1
    labeled_gtv.astype(int)
    structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
    seg=np.squeeze(seg)
    
    
    #labeled_gtv, ncomponents_gtv = label(gtv, structure) 
    labeled_gtv=np.squeeze(gtv) #logic for assumed single lesion
    ncomponents_gtv=1
    
    labeled_seg, ncomponents_seg = label(seg, structure) 
    
    #DSC_list=np.zeros([ncomponents_gtv,1])
    #precision_list=np.zeros([ncomponents_seg,1])
    DSC_array=np.zeros([ncomponents_seg,ncomponents_gtv])
    #print('n components seg: %i' % ncomponents_seg)
    for ilabel in range(0,ncomponents_seg):
        #print('n seg: %i' % (ilabel+1))
        for jlabel in range(0,ncomponents_gtv):
            pred=np.zeros(np.shape(seg)) 
            target=np.zeros(np.shape(seg)) 
            
            pred[labeled_seg==(ilabel+1)]=1.0
            target[labeled_gtv==(jlabel+1)]=1.0
            
            DSC_array[ilabel,jlabel]=np.sum(pred*target)*2.0 /(np.sum(pred) + np.sum(target))
            #print(DSC_array[ilabel,jlabel])
            
    final_pred=np.zeros(np.shape(seg)) 
    FD=0.0
    #print('Components: %i' %ncomponents_seg)
    for ilabel in range(0,ncomponents_seg):
        for jlabel in range(0,ncomponents_gtv):
            if DSC_array[ilabel,jlabel]>=0.1:
                #print('detection')
                #print(np.sum((labeled_seg==ilabel+1)*labeled_gtv)*2.0 /(np.sum(labeled_seg==ilabel+1) + np.sum(labeled_gtv)))
                final_pred[labeled_seg==ilabel+1]=1.0
                
            else:
                # pred=np.zeros(np.shape(seg)) 
                # pred[labeled_seg==ilabel+1]=1.0
                FD+=1.0
               
                # if np.sum(pred)>25.0:
                #     FD+=1
                # else:
                #     print("lesion size %i voxels filtered" % np.sum(pred))
                # #     print('False Positive!')
                # # else:
                #     print('Too small to count!')
                #     print(np.sum(pred))
                     
    # inter=np.sum(2*final_pred * labeled_gtv)
    # union=inter+abs(np.sum(final_pred-labeled_gtv))
    
    
    sd=HD.compute_surface_distances(final_pred.astype(bool), labeled_gtv.astype(bool), spacing_mm)
    hd95=HD.compute_robust_hausdorff(sd,95)
    
    DSC=np.sum(final_pred*labeled_gtv)*2.0 /(np.sum(final_pred) + np.sum(labeled_gtv))
    
    if hd95==float('Inf'):
        hd95=float('NaN')
        
    if DSC==0:
        DSC=float('NaN')   
       
    return  DSC, FD, hd95      
        # if np.sum(labeled_seg==ilabel)<27:
        #     seg[labeled_seg==ilabel]=


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

model_path = os.path.join(root_dir,opt.name,'AVG_best_finetuned') #load weights

print('loading %s' %model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_MR_seg_A(model_path) #use weights


#Prep model for eval
for m in model.netSeg_A.modules():
    for child in m.children():
        if type(child) == torch.nn.BatchNorm2d:
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None







#model.to(device)

#freeze all layers
#for param in model.netSeg_A.parameters():
#        param.requires_grad = False

                
#impath = os.path.join(path,'nii_vols') #load weights
#impath = os.path.join(root_dir,'training_data') #load weights
valpath = os.path.join(root_dir,opt.data_folder) #load weights
#valpath = os.path.join(root_dir,'test_data') #load weights


#unfreeze final layer and Channel input block for finetuning
# model.netSeg_A.CNN_block1.requires_grad_(True)
# model.netSeg_A.out_conv.requires_grad_(True)
# model.netSeg_A.out_conv1.requires_grad_(True)
# if opt.deeplayer>0:
#     model.netSeg_A.deepsupconv.requires_grad_(True)
dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(root_dir,opt.name,'seg_test_DSC.csv')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
fd_results = open(wt_path, 'w')
fd_results.write('Filename, Lesion DSC, whole volume DSC, hd95 (mm) \n')
#fd_results.write(img_name + ',' + str(DSC) +','+ str(dice_3D_temp) + ',' + str(hd95) +'\n' )
    


# results_path=os.path.join(root_dir,opt.name,'val_results')
# if not os.path.exists(results_path):
#     os.makedirs(results_path)   
seg_path=os.path.join(root_dir,opt.name,'test_segs')
if not os.path.exists(seg_path):
    os.makedirs(seg_path)    
    
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



# types = ('ProstateX*_ep2d_diff_*.nii.gz', 'MSK_MR_*_ADC.nii.gz') # the tuple of file types
# images=[]
# for fname in types:
#    images.extend(glob(os.path.join(impath, fname)))
# images = sorted(images)

# types = ('ProstateX*Finding*t2_tse_tra*_ROI.nii.gz', 'MSK_MR_*_GTV.nii.gz') # the tuple of file types
# segs=[]
# for fname in types:
#    segs.extend(glob(os.path.join(impath, fname)))
# segs = sorted(segs)
   
# types = ('ProstateX*_t2_tse*.nii.gz', 'MSK_MR_*T2w.nii.gz') # the tuple of file types
# images_t2w=[]
# for fname in types:
#    images_t2w.extend(glob(os.path.join(impath, fname)))
# images_t2w = sorted(images_t2w)




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




types = ('ProstateX*_ep2d_diff_*.nii.gz', 'MSK_MR_*_ADC.nii.gz', 'MSK_SBRT_*BL_ADC.nii','ADC_*.nii.gz','DA_*_diffusion_*.nii.gz') # the tuple of file types
val_images=[]
for fname in types:
   val_images.extend(glob(os.path.join(valpath, fname)))
val_images = sorted(val_images)

types = ('ProstateX*Finding*t2_tse_tra*_ROI.nii.gz', 'MSK_MR_*_GTV.nii.gz', 'MSK_SBRT_*BL_mask.nii','label_*.nii.gz','DA_*GTV_DIL*.nii.gz') # the tuple of file types
val_segs=[]
for fname in types:
   val_segs.extend(glob(os.path.join(valpath, fname)))
val_segs = sorted(val_segs)
 

types = ('ProstateX*_t2_tse*.nii.gz', 'MSK_MR_*T2w.nii.gz', 'MSK_SBRT_*BL_T2.nii','T2_*.nii.gz','DA_*_image*.nii.gz') # the tuple of file types
val_images_t2w=[]
for fname in types:
   val_images_t2w.extend(glob(os.path.join(valpath, fname)))
val_images_t2w = sorted(val_images_t2w)

types = ('ProstateX-????.nii.gz', 'MSK_MR_*_CTV.nii.gz', 'MSK_SBRT_*BL_CTV.nii*','CTV_*.nii.gz','DA_*_CTV_PROST*.nii.gz') # the tuple of file types
val_prost=[]
for fname in types:
    val_prost.extend(glob(os.path.join(valpath, fname)))
val_prost = sorted(val_prost)




# print(images)
# print(segs)
# print(images_t2w)

print(val_images)
print(val_segs)
print(val_images_t2w)
print(val_prost)


val_files = [{"img": img, "seg": seg, "t2w": t2w, "prost": prost} for img, seg, t2w, prost in zip(val_images, val_segs, val_images_t2w, val_prost)] #last  n_val to validation

#print(val_files)

val_transforms = Compose(
    [
        LoadImaged(keys=["img","t2w","seg","prost"]),
        EnsureChannelFirstd(keys=["img","t2w", "seg","prost"]),
        Orientationd(keys=["img","t2w","seg","prost"], axcodes="RAS"),     
        
        ResampleToMatchd(keys=["img","seg","prost"],
                              key_dst="t2w",
                              mode=("bilinear", "nearest","nearest")),
        Spacingd(keys=["img", "t2w", "seg","prost"],
                      pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                      mode=("bilinear", "bilinear", "nearest","nearest")),
        
        ScaleIntensityRangePercentilesd(keys=["img","t2w"],lower=0,upper=99,b_min=-1.0,b_max=1.0,clip=True),
        CropForegroundd(keys=["img","t2w","seg","prost"], source_key= "prost", margin=[96,96,opt.extra_neg_slices+(opt.nslices-1)/2]),
        EnsureTyped(keys=["img","t2w", "seg","prost"]),
    ]
)

post_transforms = Compose([
        EnsureTyped(keys=["pred","prost","seg"]),
        Invertd(
            keys=["pred","prost","seg"],
            transform=val_transforms,
            orig_keys="t2w",
            meta_keys=["pred_meta_dict","pred_meta_dict","pred_meta_dict"],
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys=["pred","prost","seg"], argmax=False),

    ])


# 3D dataset with preprocessing transforms
# train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
# #train_ds = monai.data.ShuffleBuffer(patch_ds, seed=0) #patch based trainer



val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)


# train_loader = DataLoader(
#     train_ds,
#     batch_size=opt.batchSize,
#     num_workers=1,
#     pin_memory=torch.cuda.is_available(),
#     drop_last=True
# )
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)


check_data = monai.utils.misc.first(val_loader)
print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

sitk.WriteImage(sitk.GetImageFromArray(check_data["t2w"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_t2w.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["img"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_adc.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["seg"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_SEG.nii.gz'))
sitk.WriteImage(sitk.GetImageFromArray(check_data["prost"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_PROST.nii.gz'))



num_epochs=opt.niter+opt.niter_decay
epoch_loss_values = []
best_dice=0
mixup_ab=opt.mixup_betadist

with torch.no_grad(): # no grade calculation 
    dice_3D=[]
    Lesion_Dice=[]
    hd95_list=[]
    False_positives=[]
    smooth=1
    step=0
    for val_data in val_loader:
        step += 1
        adc, label_val,t2w,prostate = val_data["img"].to(device), val_data["seg"].to(device), val_data["t2w"].to(device), val_data["prost"]
        
        adc_name=adc.meta['filename_or_obj'][0].split('/')[-1]
        t2_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
        gtv_name=label_val.meta['filename_or_obj'][0].split('/')[-1]
        
        print('  T2: %s ADC: %s GTV: %s' % (t2_name, adc_name, gtv_name))



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
            

        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
                        # seg_out= from_engine(["pred"])(vol_data)[0]
                        # seg_out = np.array(seg_out)
                        # seg_out=np.squeeze(seg_out)
                        # seg_out[seg_out >= 0.5]=1.0
                        # seg_out[seg_out < 0.5]=0.0
                        # seg_out = np.transpose(seg_out, (2, 1, 0))
                        
        seg = from_engine(["pred"])(val_data)[0]
        gtv = from_engine(["seg"])(val_data)[0]
        prostate = from_engine(["prost"])(val_data)[0]
        
        
        
        #print("seg length: ", len(seg))
        # prostate=prostate[0]
        # seg = seg[0]
        
        
        #print("seg shape: ", np.shape(seg))
        gtv = np.array(gtv)
        gtv=np.squeeze(gtv)
        
        if  np.sum(gtv)>5:
            
            prostate = np.array(prostate)
            prostate=np.squeeze(prostate)
            prostate=ndimage.binary_dilation(prostate).astype(prostate.dtype)
            if np.sum(prostate*gtv)<(0.9*np.sum(gtv)):
                prostate[prostate<1.0]=1.0
                print('prostate mask issue')
            
            seg = np.array(seg)
            seg=np.squeeze(seg)
            #print('seg sum:', np.sum(seg) )
            #print('seg max:', np.max(seg) )
            
            seg[seg >= 0.5]=1.0
            seg[seg < 0.5]=0.0
            
            #print('seg sum2:', np.sum(seg) )
            seg_filtered= np.array(seg)
            
            
            seg_filtered[prostate < 0.5]=0.0
            
            # #filter out small unconnected segs @0.5x0.5x3 27 vox ~ 2 mL
            # structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
            # labeled, ncomponents = label(seg_filtered, structure)    
            # for ilabel in range(0,ncomponents+1):
            #     if np.sum(labeled==ilabel+1)<27:
            #         seg_filtered[labeled==ilabel]=0
    
           
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
            #seg = np.array(seg)
            #prostate= from_engine(["prost"])(val_data)[0]
            #prostate = np.array(prostate)
            #prostate=np.squeeze(prostate)
            
            
            #seg[seg >= 0.5]=1.0
            #seg[seg < 0.5]=0.0
            
            # seg_filtered= np.array(seg)
            #seg_filtered[prostate < 0.5]=0.0
            img_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
            cur_rd_path=os.path.join(valpath,img_name)
            im_obj = sitk.ReadImage(cur_rd_path)
            
            spacing_mm=im_obj.GetSpacing()
            
            seg_temp=np.array(seg_filtered)
            #gt_temp=np.array(label_val_vol.cpu())
            
            gtv=np.squeeze(gtv)
            
            # print(np.shape(seg_temp))
            # print(np.shape(gtv))
            
            seg_flt=seg_temp.flatten()
            gt_flt=gtv.flatten()
            
            # print(img_name)
            
            intersection = np.sum(seg_flt * (gt_flt > 0))
            dice_3D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
            print('  Volumetric DSC: %f'  %dice_3D_temp)
            dice_3D.append(dice_3D_temp)
            
            DSC,FP,hd95=lesion_eval(seg_filtered,gtv,spacing_mm)
            Lesion_Dice.append(DSC)
            hd95_list.append(hd95)
            False_positives.append(FP)
            print('  Lesion DSC: %f' %DSC)
            print('  Lesion HD95: %f mm' %hd95)
            #print('FP: %i' %FP)
            
            
            fd_results.write(img_name + ',' + str(DSC) +','+ str(dice_3D_temp) + ',' + str(hd95) +'\n' )
    
            
            seg = np.transpose(seg, (2, 1, 0))
            seg_filtered = np.transpose(seg_filtered, (2, 1, 0))
            prostate = np.transpose(prostate, (2, 1, 0))
            
            # img_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
            
            
            
            seg = sitk.GetImageFromArray(seg)
            seg = copy_info(im_obj, seg)
            sitk.WriteImage(seg,  os.path.join(seg_path,'seg_%s' % img_name))
            
            
            seg_filtered = sitk.GetImageFromArray(seg_filtered)
            seg_filtered = copy_info(im_obj, seg_filtered)
            sitk.WriteImage(seg_filtered,  os.path.join(seg_path,'filteredseg_%s' % img_name))
            
            
            prostate = sitk.GetImageFromArray(prostate)
            prostate = copy_info(im_obj, prostate)
            sitk.WriteImage(prostate,  os.path.join(seg_path,'prostateseg_%s' % img_name))
        
            #model.save('AVG_best_finetuned')
    dice_3D=np.median(dice_3D)
    median_Lesion_Dice=np.nanmedian(Lesion_Dice)
    median_hd95=np.nanmedian(hd95_list)
    
    print('Median whole volume dice: %f' % dice_3D)
    print('Median Lesion dice: %f' % median_Lesion_Dice)
    #print('mean HD95: %f' % mean_hd95)
    print('median Lesion HD95: %f' % median_hd95)
     
    # print(Lesion_Dice)
    # print(FP)
    
    # print(np.sum(np.float(np.array(Lesion_Dice)>0.1)))
    # print(np.float(np.size(Lesion_Dice)))
    
    
    recall=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice))
    #print(recall)
    print('recall: %f' % recall)
    
    precision=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice)+np.sum(FP))
    
    # print(np.sum(np.float(np.array(Lesion_Dice)>0.1)))
    # print(np.float(np.size(Lesion_Dice)+np.sum(FP)))
    
    
    # print(precision)
    print('precision: %f' % precision)
    #print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D, ' (best: %.2f)'  % best_dice)
    fd_results.flush()  
    
    wt2_path= os.path.join(root_dir,opt.name,'Result_summary.csv')
    fd_results_sum = open(wt2_path, 'w')
    fd_results_sum.write('name, median Lesion DSC, median whole volume DSC, median lesion hd95 (mm), precision, recall \n')
    fd_results_sum.write(opt.name + ',' + str(median_Lesion_Dice) +','+ str(dice_3D) + ',' + str(median_hd95) + ',' + str(precision) + ',' + str(recall) +'\n')
    fd_results_sum.flush()  
    # if lr>0:
    #         model.update_learning_rate()