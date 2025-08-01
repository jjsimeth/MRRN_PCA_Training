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
from scipy.ndimage import distance_transform_edt
import re
from collections import defaultdict

print(torch.version.cuda)

def print_metrics(DSC, hd95, dice_3D_temp, gt_vol, pred_vol):
    # Helper to format values (handle list or scalar)
    def format_val(val):
        if isinstance(val, (list, np.ndarray)):
            return ', '.join(f'{v:.3f}' for v in val)
        else:
            return f'{val:.3f}'

    print(f'  Lesion DSC: {format_val(DSC)}')
    print(f'  Lesion HD95: {format_val(hd95)} mm')
    print(f'  Volumetric DSC: {format_val(dice_3D_temp)}')
    print(f'  Lesion vol: {format_val(gt_vol)} mL')
    print(f'  Prediction vol: {format_val(pred_vol)} mL')
    
def combine_lists_with_sources(*lists):
    combined = []
    sources = []
    for idx, lst in enumerate(lists):
        combined.extend(lst)
        sources.extend([idx] * len(lst))
    return combined, sources

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

def relabel_splits_and_merges(labeled_gtv, ncomponents_gtv,
                              labeled_seg, ncomponents_seg,
                              DSC_array):
    new_pred = np.zeros_like(labeled_seg, dtype=np.int32)
    current_label = 1
    
    # For bookkeeping
    pred_to_gt = [set(np.where(DSC_array[i, :] > 0)[0] + 1) for i in range(ncomponents_seg)]
    gt_to_pred = [set(np.where(DSC_array[:, j] > 0)[0] + 1) for j in range(ncomponents_gtv)]
    
    new_labels_list = []  # will store tuples like ('merge', gt_id, [pred_ids]) or ('split', pred_id, [gt_ids]) or ('single', pred_id, gt_id)
    
    # 1. Handle merges (multiple preds correspond to one gt)
    for gt_id in range(1, ncomponents_gtv + 1):
        preds_for_gt = gt_to_pred[gt_id - 1]
        if len(preds_for_gt) == 0:
            continue
        elif len(preds_for_gt) == 1:
            pred_id = list(preds_for_gt)[0]
            new_pred[labeled_seg == pred_id] = current_label
            new_labels_list.append(('single', pred_id, gt_id, current_label))
            current_label += 1
        else:
            merged_mask = np.isin(labeled_seg, list(preds_for_gt))
            new_pred[merged_mask] = current_label
            new_labels_list.append(('merge', gt_id, list(preds_for_gt), current_label))
            current_label += 1
    
    # 2. Handle splits (one pred corresponds to multiple gts)
    for pred_id in range(1, ncomponents_seg + 1):
        gts_for_pred = pred_to_gt[pred_id - 1]
        if len(gts_for_pred) <= 1:
            continue
        
        pred_mask = (labeled_seg == pred_id)
        coords = np.argwhere(pred_mask)
        
        dist_maps = []
        for gt_id in gts_for_pred:
            gt_mask = (labeled_gtv == gt_id)
            dist_maps.append(distance_transform_edt(~gt_mask))
        
        dist_maps = np.array(dist_maps)  # shape (num_gt, D, H, W)
        
        dist_vals = np.array([dist_maps[i][tuple(coord)] for i in range(len(gts_for_pred)) for coord in coords])
        dist_vals = dist_vals.reshape(len(gts_for_pred), len(coords)).T
        
        closest_gt_indices = np.argmin(dist_vals, axis=1)
        
        for i, gt_idx in enumerate(gts_for_pred):
            voxels = coords[closest_gt_indices == i]
            for v in voxels:
                new_pred[tuple(v)] = current_label
            new_labels_list.append(('split', pred_id, gt_idx, current_label))
            current_label += 1
    
    # 3. Handle preds with no assignment
    assigned_preds = set(np.unique(new_pred))
    assigned_preds.discard(0)
    
    for pred_id in range(1, ncomponents_seg + 1):
        if pred_id not in assigned_preds:
            new_pred[labeled_seg == pred_id] = current_label
            new_labels_list.append(('unassigned', pred_id, None, current_label))
            current_label += 1
    
    return new_pred, new_labels_list

def compute_metrics_per_lesion(labeled_gtv, labeled_pred, spacing_mm):
    labels_gt = np.unique(labeled_gtv)
    labels_gt = labels_gt[labels_gt != 0]  # skip background

    DSC_list = []
    FD = 0.0
    hd95_list = []
    gt_vol_list = []
    pred_vol_list = []

    for label_gt in labels_gt:
        gt_mask = (labeled_gtv == label_gt)

        # Find overlapping predicted regions
        overlapping_pred_labels = np.unique(labeled_pred[gt_mask])
        overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels != 0]

        best_dsc = 0.0
        best_pred_label = None

        for pred_label in overlapping_pred_labels:
            pred_mask = (labeled_pred == pred_label)
            inter = np.sum(gt_mask & pred_mask)
            denom = np.sum(gt_mask) + np.sum(pred_mask)
            dsc = 2.0 * inter / denom if denom > 0 else 0.0
            if dsc > best_dsc:
                best_dsc = dsc
                best_pred_label = pred_label
           # False negative if 

        # Compute volumes
        gt_vol = np.sum(gt_mask) * np.prod(spacing_mm) / 1000.0
        pred_mask = (labeled_pred == best_pred_label) if best_pred_label is not None else np.zeros_like(gt_mask)
        pred_vol = np.sum(pred_mask) * np.prod(spacing_mm) / 1000.0

        # Compute HD95
        if best_dsc > 0:
            sd = HD.compute_surface_distances(gt_mask, pred_mask, spacing_mm)
            hd95 = HD.compute_robust_hausdorff(sd, 95)
        else:
            hd95 = float('NaN')

        DSC_list.append(best_dsc if best_dsc > 0 else float(0.0))
        hd95_list.append(hd95)
        gt_vol_list.append(gt_vol)
        pred_vol_list.append(pred_vol)
    
    FD= np.size(np.unique(labeled_pred[labeled_pred != 0]))-np.sum(np.array(DSC_list)>0.1)
    # print(np.size(np.unique(labeled_pred[labeled_pred != 0])))
    # print(np.sum(np.array(DSC_list)>0.1))
    return DSC_list, FD, hd95_list, gt_vol_list, pred_vol_list


def lesion_eval(seg,gtv,spacing_mm):
  #filter out small unconnected segs
    labeled_gtv=np.squeeze(gtv)
    labeled_gtv[labeled_gtv>0.0]=1
    labeled_gtv.astype(int)
    structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
    seg=np.squeeze(seg)
    
    
    labeled_gtv, ncomponents_gtv = label(gtv, structure)
    
    #labeled_gtv=np.squeeze(gtv) #logic for assumed single lesion
    #ncomponents_gtv=1
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
    
            
    
    labeled_seg,labels_seg=relabel_splits_and_merges(labeled_gtv, ncomponents_gtv,labeled_seg, ncomponents_seg,DSC_array)
    
    DSC, FD, hd95, gt_vol, pred_vol=compute_metrics_per_lesion(labeled_gtv, labeled_seg, spacing_mm)
    
    #print(labels_seg)
    print('  lesions/preds, FP: %i/%i, %i' % (ncomponents_gtv,np.size(np.unique(labeled_seg[labeled_seg != 0])),FD))
    return  DSC, FD, hd95, gt_vol, pred_vol    
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

if opt.modality.lower()=='adc':
    nmodalities=1
elif ('t2' in opt.modality.lower()) and ('adc' in opt.modality.lower()):
    nmodalities=2
elif opt.modality.lower()=='t2':
    nmodalities=1
else:
    print('No modality option, assuming ADC...?')
    nmodalities=1

root_dir=os.getcwd()
opt.nchannels=opt.nslices*nmodalities


models = []
weights_dir = os.path.join(root_dir, opt.name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

suffix = "_net_Seg_A.pth"

print('weights path:'+str(weights_dir))
print(weights_dir)
for filename in os.listdir(weights_dir):
    if filename.endswith(suffix):
        # Strip suffix to get the base name expected by load_MR_seg_A
        base_name = filename[:-len(suffix)]
        print(f'Loading model from base name: {base_name}')

        # Create and load model
        model = create_model(opt)
        model.load_MR_seg_A(base_name)  # pass only the base name

        # Prepare model for eval
        for m in model.netSeg_A.modules():
            for child in m.children():
                if isinstance(child, torch.nn.BatchNorm2d):
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

        model.netSeg_A.eval()
        models.append(model)

resultsdir= r'/lila/data/deasy/Josiah_data/Prostate/results/NKI_MSK'   
dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(root_dir,opt.name,'ct_seg_val_loss.csv')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
fd_results = open(wt_path, 'w')
fd_results.write('train loss, seg accuracy,\n')
dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(resultsdir,'seg_test_DSC_%s_%s_%s.csv' %(opt.model_to_test,opt.name,'P158'))
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
fd_results = open(wt_path, 'w')
fd_results.write('Filename, LesionN, Lesion DSC, whole volume DSC, hd95 (mm), gt vol (mL), pred vol(mL) \n')
#fd_results.write(img_name + ',' + str(DSC) +','+ str(dice_3D_temp) + ',' + str(hd95) +'\n' )
    


# results_path=os.path.join(root_dir,opt.name,'val_results')
# if not os.path.exists(results_path):
#     os.makedirs(results_path)   
seg_path=os.path.join(root_dir,opt.name,'test_segs')
if not os.path.exists(seg_path):
    os.makedirs(seg_path)    
    



# types = ('ProstateX*_ep2d_diff_*.nii.gz', 'MSK_MR_*_ADC.nii.gz', 'MSK_SBRT_*BL_ADC.nii','MSK_DILVAR*_ADC.nii.gz') # the tuple of file types
# val_images=[]
# for fname in types:
#    val_images.extend(glob(os.path.join(valpath, fname)))
# val_images = sorted(val_images)

# types = ('ProstateX*Finding*t2_tse_tra*_ROI.nii.gz', 'MSK_MR_*_GTV.nii.gz', 'MSK_SBRT_*BL_mask.nii','MSK_DILVAR*_GTV.nii.gz') # the tuple of file types
# val_segs=[]
# for fname in types:
#    val_segs.extend(glob(os.path.join(valpath, fname)))
# val_segs = sorted(val_segs)
 

# types = ('ProstateX*_t2_tse*.nii.gz', 'MSK_MR_*T2w.nii.gz', 'MSK_SBRT_*BL_T2.nii','MSK_DILVAR*_T2.nii.gz') # the tuple of file types
# val_images_t2w=[]
# for fname in types:
#     val_images_t2w.extend(glob(os.path.join(valpath, fname)))
# val_images_t2w = sorted(val_images_t2w)

# types = ('ProstateX-????.nii.gz', 'MSK_MR_*_CTV.nii.gz', 'MSK_SBRT_*BL_CTV.nii*') # the tuple of file types
# val_prost=[]
# for fname in types:
#     val_prost.extend(glob(os.path.join(valpath, fname)))
# val_prost = sorted(val_prost)


#opt.extra_neg_slices=5

# print(images)
# print(segs)
# print(images_t2w)

# print(val_images)
# print(val_segs)
# print(val_images_t2w)
# print(val_prost)



# datadir= r'/lila/data/deasy/Josiah_data/Prostate/nii_data'   

# valpath_adcpath = os.path.join(datadir,'MR_ProstateX','Images','ADC')
# valpath_t2path = os.path.join(datadir,'MR_ProstateX','Images','T2w')
# valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','ADC_GS')
# #valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','T2w_GS')
# valpath_prostate_masks = os.path.join(datadir,'MR_ProstateX','Masks','Prostate')

# impath=valpath_adcpath

# valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
# custom_names = ["img", "seg", "t2w","prost"]

# val_files = find_common_files(valfolders, custom_names,pattern=r'.*ProstateX-(\d+).*\.nii\.gz')




datapaths = [r'/lila/data/deasy/Josiah_data/Prostate/nii_data', r'/gpfs/home1/rbosschaert/']
datadir = next((path for path in datapaths if os.path.exists(path)), None)  

resultspaths= [r'/lila/data/deasy/Josiah_data/Prostate/results/NKI_MSK' , r'/gpfs/home1/rbosschaert/NKI_MSK']
resultsdir = next((path for path in resultspaths if os.path.exists(path)), None)  


if opt.test_case.lower()=='prostate158':
    valpath_adcpath = os.path.join(datadir,'MR_Prostate158_Train','Images','ADC')
    valpath_t2path = os.path.join(datadir,'MR_Prostate158_Train','Images','T2w')
    valpath_masks = os.path.join(datadir,'MR_Prostate158_Train','Masks','DIL')
    valpath_masks2 = os.path.join(datadir,'MR_Prostate158_Train','Masks','DIL2')
    valpath_prostate_masks = os.path.join(datadir,'MR_Prostate158_Train','Masks','Prostate')
    
    impath=valpath_adcpath
    
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files1 = find_common_files(valfolders, custom_names,pattern=r'.*_P(\d+)_.*\.nii\.gz$')
    
    valfolders = [valpath_adcpath, valpath_masks2,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files2 = find_common_files(valfolders, custom_names,pattern=r'.*_P(\d+)_.*\.nii\.gz$')
    
    val_files,Allgroups=combine_lists_with_sources(val_files1,val_files2)
    
elif opt.test_case.lower()=='prostatex':  
    valpath_adcpath = os.path.join(datadir,'MR_ProstateX','Images','ADC')
    valpath_t2path = os.path.join(datadir,'MR_ProstateX','Images','T2w')
    valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','ADC_GS')
    #valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','T2w_GS')
    valpath_prostate_masks = os.path.join(datadir,'MR_ProstateX','Masks','Prostate')
    
    impath=valpath_adcpath
    
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files = find_common_files(valfolders, custom_names,pattern=r'.*ProstateX-(\d+).*\.nii\.gz')    
elif opt.test_case.lower()=='msk':  
    valpath_adcpath = os.path.join(datadir,'LINAC95_B_nii','Images','ADC')
    valpath_t2path = os.path.join(datadir,'LINAC95_B_nii','Images','T2w')
    valpath_masks = os.path.join(datadir,'LINAC95_B_nii','Masks','DIL')
    valpath_prostate_masks = os.path.join(datadir,'LINAC95_B_nii','Masks','Prostate')
    
    impath=valpath_adcpath
    
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files = find_common_files(valfolders, custom_names,pattern=r'.*_P(\d+)_S(\d+).*\.nii\.gz')
  



# print("Matching files:")
# for file_entry in val_files:
#      print(file_entry)
    

val_transforms = Compose(
    [
        LoadImaged(keys=["img","seg","prost", "t2w"]),
        EnsureChannelFirstd(keys=["img", "seg","prost", "t2w"]),
        Orientationd(keys=["img","seg","prost", "t2w"], axcodes="RAS"),     
        
        ResampleToMatchd(keys=["t2w","seg","prost"],
                              key_dst="img",
                              mode=("bilinear", "nearest", "nearest")),
        Spacingd(keys=["img", "seg","prost", "t2w"],
                      pixdim=(PD_in[0], PD_in[1], PD_in[2]),
                      mode=("bilinear", "nearest", "nearest","bilinear")),
        
        ScaleIntensityRangePercentilesd(keys=["img", "t2w"],lower=0,upper=99,b_min=-1.0,b_max=1.0,clip=True,channel_wise=True),
        #CenterSpatialCropd(keys=["img","t2w","prost","dose","seg"], roi_size=[256, 256,-1]),
        
        # CropForegroundd(keys=["img","seg","prost", "t2w"], source_key= "seg", margin=[128,128,opt.extra_neg_slices+(opt.nslices-1)/2]),
        # CropForegroundd(keys=["img","seg","prost", "t2w"], source_key= "prost", margin=[64,64,0]),
        
        #CropForegroundd(keys=["img", "t2w", "prost", "seg"], source_key="seg", margin=[256, 256, opt.extra_neg_slices + (opt.nslices - 1) / 2],allow_smaller=True),
        CropForegroundd(keys=["img","seg","prost", "t2w"], source_key= "prost", margin=[64,64,opt.extra_neg_slices + (opt.nslices - 1) / 2]),
        
        
        #RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),,
        CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[128, 128,-1]),
        EnsureTyped(keys=["img", "seg","prost", "t2w"]),
        AsDiscreted(keys=["prost","seg"], rounding="torchrounding"),
    ]
  )

post_transforms = Compose([
        EnsureTyped(keys=["pred","seg","prost"]),
        Invertd(
            keys=["pred","seg","prost"],
            transform=val_transforms,
            orig_keys="img",
            meta_keys=["pred_meta_dict","pred_meta_dict","pred_meta_dict"],
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys=["pred","seg","prost"], argmax=False),

    ])



# 3D dataset with preprocessing transforms
# train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
# #train_ds = monai.data.ShuffleBuffer(patch_ds, seed=0) #patch based trainer




#val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

# train_loader = DataLoader(
#     train_ds,
#     batch_size=opt.batchSize,
#     num_workers=1,
#     pin_memory=torch.cuda.is_available(),
#     drop_last=True
# )



# check_data = monai.utils.misc.first(val_loader)
# print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

# sitk.WriteImage(sitk.GetImageFromArray(check_data["t2w"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_t2w.nii.gz'))
# sitk.WriteImage(sitk.GetImageFromArray(check_data["img"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_adc.nii.gz'))
# sitk.WriteImage(sitk.GetImageFromArray(check_data["seg"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_SEG.nii.gz'))
# sitk.WriteImage(sitk.GetImageFromArray(check_data["prost"].cpu()[:,:,:,:]),  os.path.join(dest_path,'TRAINING_DEBUG_PROST.nii.gz'))



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
        adc, label_val,prostate,t2w = val_data["img"].to(device), val_data["seg"].to(device), val_data["prost"], val_data["t2w"].to(device)
        

        label_val_vol=label_val
        #if torch.sum(label_val)>0:
        # adc=scale_ADC(adc)
        
        if opt.modality.lower()=='adc':
            val_inputs=adc
        elif ('t2' in opt.modality.lower()) and ('adc' in opt.modality.lower()) and ('reverse' in opt.modality.lower()):
            val_inputs=torch.cat((t2w,adc),dim=1)    
        elif ('t2' in opt.modality.lower()) and ('adc' in opt.modality.lower()):
            val_inputs=torch.cat((adc,t2w),dim=1)
        
        elif opt.modality.lower()=='t2':
            val_inputs=t2w
        else:
            print('No modality option, assuming ADC...?')
            val_inputs=adc    
       
        adc_name=adc.meta['filename_or_obj'][0].split('/')[-1]
        t2_name=t2w.meta['filename_or_obj'][0].split('/')[-1]
        gtv_name=label_val.meta['filename_or_obj'][0].split('/')[-1]
        #ktrans_name=ktrans.meta['filename_or_obj'][0].split('/')[-1]
        prostate_name=prostate.meta['filename_or_obj'][0].split('/')[-1]
        
        print('ADC: %s T2w: %s GTV: %s Prostate: %s' % (adc_name,t2_name,gtv_name,prostate_name))
       
        adc_name=adc.meta['filename_or_obj'][0]
        t2_name=t2w.meta['filename_or_obj'][0]
        gtv_name=label_val.meta['filename_or_obj'][0]
        #ktrans_name=ktrans.meta['filename_or_obj'][0]
        prostate_name=prostate.meta['filename_or_obj'][0]
        
        # print('******************************************************')
        # print('  ADC: %s'  % (adc_name))
        # print('  T2w: %s'  % (t2_name))
        # #print('  Ktrans: %s'  % (ktrans_name))
        # print('  GTV: %s'  % (gtv_name))
        # print('  Prostate: %s'  % (prostate_name))
        
        print('******************************************************')
        #print('  ADC: %s'  % os.path.basename(adc_name), 'T2w: %s'  % os.path.basename(t2_name), 'GTV: %s'  % os.path.basename(gtv_name),'Prostate: %s'  % os.path.basename(prostate_name) )

        
        predictions=[]
        #print('models: %i' %len(models))
        with torch.no_grad():
            with autocast(enabled=True):
                for model in models:
                    # sliding_window_inference returns a torch.Tensor
                    pred = sliding_window_inference(
                        val_inputs,                       # e.g. (B,C,H,W,D)
                        (128, 128, opt.nslices),         # ROI size
                        1,                               # sw_batch_size
                        model.netSeg_A,
                        overlap=0.80,
                        mode="gaussian",
                        sigma_scale=[0.128, 0.128, 0.001],
                        options=opt
                    )
                    # ensure float precision
                    predictions.append(pred)
                    
        # if you really need a NumPy array:
        val_data["pred"] = torch.mean(torch.stack(predictions), dim=0)
            
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
        
        
        # print("seg shape: ", np.shape(seg))
        # print("gtv shape: ", np.shape(gtv))
        # print("prostate shape: ", np.shape(prostate))
        
        gtv = np.array(gtv)
        gtv=np.squeeze(gtv)
        
        if  np.sum(gtv)>5:
            
            prostate = np.array(prostate)
            prostate=np.squeeze(prostate)
            prostate[prostate>0.5]=1.0
            prostate[prostate<1.0]=0.0
            
            gtv[gtv>0.5]=1.0
            gtv[gtv<1.0]=0.0
            
            prostate=ndimage.binary_dilation(prostate).astype(prostate.dtype)
            if np.sum(prostate*gtv)<(0.9*np.sum(gtv)):
                print('prostate mask issue')
                prostate[prostate<1.0]=1.0
                
            seg = np.array(seg)
            seg=np.squeeze(seg)
            #print('seg sum:', np.sum(seg) )
            #print('seg max:', np.max(seg) )
            
            seg[seg >= opt.seg_threshold]=1.0
            seg[seg < opt.seg_threshold]=0.0

            
            #print('seg sum2:', np.sum(seg) )
            seg_filtered= np.array(seg)
            
            
            seg_filtered[prostate < 1.0]=0.0
            
            # #filter out small unconnected segs @0.5x0.5x3 27 vox ~ 2 mL
            structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
            labeled, ncomponents = label(seg_filtered, structure)    
            # for ilabel in range(0,ncomponents+1):
            #     if np.sum(labeled==ilabel+1)<15:
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
            img_name=adc.meta['filename_or_obj'][0].split('/')[-1]
            cur_rd_path=os.path.join(impath,img_name)
            im_obj = sitk.ReadImage(cur_rd_path)
            
            spacing_mm=im_obj.GetSpacing()
            
            seg_temp=np.array(seg_filtered)
            #gt_temp=np.array(label_val_vol.cpu())
            
            gtv=np.squeeze(gtv)
            
            # print(np.shape(seg_temp))
            # print(np.shape(gtv))
            
            seg_flt=seg_temp.flatten()
            gt_flt=gtv.flatten()
            gt_flt=gt_flt > 0.5
            #print(img_name)
            
            intersection = np.sum(seg_flt * (gt_flt > 0))
            dice_3D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
            dice_3D.append(dice_3D_temp)
            
            DSC,FP,hd95, gt_vol, pred_vol=lesion_eval(seg_filtered,gtv,spacing_mm)
            Lesion_Dice.extend(DSC)
            hd95_list.extend(hd95)
            False_positives.append(FP)
            # print('  Lesion DSC: %f, Lesion HD95: %f mm, Volumetric DSC: %f' %(DSC, hd95, dice_3D_temp))
            # print('  Lesion vol: %f mL, prediction vol: %f mL' %(gt_vol,pred_vol))
            print_metrics(DSC, hd95, dice_3D_temp, gt_vol, pred_vol)
            
            
            
            for ilesion in range(0,np.size(DSC)):
                fd_results.write(img_name + ',' + str(ilesion+1)+ ',' + str(DSC[ilesion]) +','+ str(dice_3D_temp) + ',' + str(hd95[ilesion]) + ',' + str(gt_vol[ilesion]) + ',' + str(pred_vol[ilesion]) +'\n' )
    
            
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
    print('Median num of FP per case: %i' %np.median(False_positives))
    median_dice_3D = np.nanmedian(dice_3D)
    q25_dice_3D, q75_dice_3D = np.nanpercentile(dice_3D, [25, 75])
    
    # Lesion dice for detected lesions
    Lesion_Dice=np.array(Lesion_Dice)
    median_Lesion_Dice = np.nanmedian(Lesion_Dice[Lesion_Dice>0.1])
    q25_Lesion_Dice, q75_Lesion_Dice = np.nanpercentile(Lesion_Dice, [25, 75])
    
    # Lesion HD95
    median_hd95 = np.nanmedian(hd95_list)
    q25_hd95, q75_hd95 = np.nanpercentile(hd95_list, [25, 75])
    
    # Output with formatting
    print('Whole volume Dice - Median: %.2f, IQR: %.2f–%.2f' % (median_dice_3D, q25_dice_3D, q75_dice_3D))
    print('Lesion Dice - Median: %.2f, IQR: %.2f–%.2f' % (median_Lesion_Dice, q25_Lesion_Dice, q75_Lesion_Dice))
    print('Lesion HD95 - Median: %.2f, IQR: %.2f–%.2f' % (median_hd95, q25_hd95, q75_hd95))
         
    # print(Lesion_Dice)
    # print(FP)
    
    # print(np.sum(np.float(np.array(Lesion_Dice)>0.1)))
    # print(np.float(np.size(Lesion_Dice)))
    
    
    recall=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice))
    #print(recall)
    print('recall: %f' % recall)
    
    precision=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice)+np.sum(False_positives))
    
    # print(np.sum(np.float(np.array(Lesion_Dice)>0.1)))
    # print(np.float(np.size(Lesion_Dice)+np.sum(FP)))
    
    
    # print(precision)
    print('precision: %f' % precision)
    #print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D, ' (best: %.2f)'  % best_dice)
    fd_results.flush()  
    
    wt2_path= os.path.join(resultsdir,'Result_summary_%s_%s_%s.csv' %(opt.model_to_test,opt.name,'P158'))
    fd_results_sum = open(wt2_path, 'w')
    fd_results_sum.write('name, median Lesion DSC, median whole volume DSC, median lesion hd95 (mm), precision, recall \n')
    fd_results_sum.write(opt.name + ',' + str(median_Lesion_Dice) +','+ str(dice_3D) + ',' + str(median_hd95) + ',' + str(precision) + ',' + str(recall) +'\n')
    fd_results_sum.flush()  
    # if lr>0:
    #         model.update_learning_rate()