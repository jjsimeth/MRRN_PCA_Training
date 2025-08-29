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
    ScaleIntensityRangePercentiles,
)

from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt
import re
from collections import defaultdict
from monai.transforms import MapTransform
from typing import Mapping, Hashable, Sequence, Union
import pandas as pd

print(torch.version.cuda)

class EnsureMaskExistsd(MapTransform):
    """
    If `mask` key is missing, creates a zero-valued mask with the same shape as `reference`.
    """
    def __init__(
        self,
        mask_key: Union[str, Hashable],
        reference_key: Union[str, Hashable],
    ):
        super().__init__(keys=[mask_key, reference_key])
        self.mask_key = mask_key
        self.reference_key = reference_key

    def __call__(self, data: Mapping):
        d = dict(data)
        if self.mask_key not in d:
            ref = d[self.reference_key]
            d[self.mask_key] = np.ones_like(ref, dtype=np.uint8)
        return d
    
def getDICE(pred,ref):
    
    pred[pred>0.5]=1.0
    pred[pred<1.0]=0.0
    
    ref[ref>0.5]=1.0
    ref[ref<1.0]=0.0
    
    DSC=np.sum(pred*ref)*2.0 /(np.sum(pred) + np.sum(ref))
            
    return DSC

def process_with_csv(img_name,gt_name,model_name,dataset_name, seg_filtered, gtv, adc, spacing_mm, fd_results, csv_path=None):
    # Optional CSV load
    df = None
    if csv_path is not None:
        try:
            if csv_path.lower().endswith(".csv"):
                df = pd.read_csv(csv_path, encoding="utf-8")
            elif csv_path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp1252")  # Fallback for Windows CSV

    # Enable connected component analysis only if CSV provided
    connected_component_analysis = csv_path is not None

    # Run lesion evaluation
    DSC, FP, hd95, gt_vol, pred_vol,gt_adc, pred_adc, labels = lesion_eval(
        seg_filtered, gtv, adc, spacing_mm, connected_component_analysis
    )

    Lesion_Dice.extend(DSC)
    hd95_list.extend(hd95)
    False_positives.append(FP)

    dice_3D_temp = getDICE(seg_filtered, gtv)  # Assuming you have this function
    print_metrics(DSC, hd95, dice_3D_temp, gt_vol, pred_vol)

    # Prepare patient info match from CSV
    csv_info = pd.DataFrame()
    if df is not None:
        # Extract ID from filename: at most 2 letters before, at least 3 digits, at most 1 letter after
        img_id_match = re.search(r'([A-Za-z]{0,2}\d{3,}[A-Za-z]{0,1})', img_name)
        img_id = img_id_match.group(1) if img_id_match else None
    
        if img_id:
            # Extract same style ID from CSV's Patient ID column
            df['alnum_id'] = df['Patient ID'].astype(str).str.extract(r'([A-Za-z]{0,2}\d{3,}[A-Za-z]{0,1})')
            csv_info = df[df['alnum_id'] == img_id]
            
    # Write per-lesion results
    for i, ilesion in enumerate(labels):  # i = index into metric arrays, ilesion = actual label ID
        if csv_info.empty:
            # No match or no CSV
            lesion_row = {
                "Gleason_Grade_Group": np.nan,
                "PIRADS": np.nan,
                "ISUP": np.nan,
                "Zone": np.nan,
                "Clinically_significant": np.nan,
                "ROI_Volume": np.nan,
                "Scanner": np.nan,
                "Voxel_size": np.nan
            }
        else:
            lesion_data = csv_info[csv_info["Label"] == ilesion]
            if not lesion_data.empty:
                lesion_row = {
                    "Gleason_Grade_Group": lesion_data.get("Gleason_Grade_Group", pd.Series([np.nan])).values[0],
                    "PIRADS": lesion_data.get("PIRADS", pd.Series([np.nan])).values[0],
                    "ISUP": lesion_data.get("ISUP", pd.Series([np.nan])).values[0],
                    "Zone": lesion_data.get("Zone", pd.Series([np.nan])).values[0],
                    "Clinically_significant": lesion_data.get("Clinically_significant", pd.Series([np.nan])).values[0],
                    "ROI_Volume": lesion_data.get("ROI_Volume", pd.Series([np.nan])).values[0],
                    "Scanner": lesion_data.get("Scanner", pd.Series([np.nan])).values[0],
                    "Voxel_size": lesion_data.get("Voxel_size", pd.Series([np.nan])).values[0]
                }
            else:
                lesion_row = {
                    "Gleason_Grade_Group": np.nan,
                    "PIRADS": np.nan,
                    "ISUP": np.nan,
                    "Zone": np.nan,
                    "Clinically_significant": np.nan,
                    "ROI_Volume": np.nan,
                    "Scanner": np.nan,
                    "Voxel_size": np.nan
                }
    
        # Write row: sequential lesion number (i+1), actual label ID (ilesion), metrics, CSV info
        fd_results.write(
            f"{model_name},{dataset_name},{img_name},{gt_name},{ilesion},{DSC[i]},{dice_3D_temp},{hd95[i]},"
            f"{gt_vol[i]},{pred_vol[i]},{gt_adc[i]},{pred_adc[i]},{FP},"
            f"{lesion_row['Gleason_Grade_Group']},{lesion_row['PIRADS']},{lesion_row['ISUP']},"
            f"{lesion_row['Zone']},{lesion_row['Clinically_significant']},"
            f"{lesion_row['ROI_Volume']},{lesion_row['Scanner']},{lesion_row['Voxel_size']}\n"
        )
        #model_name,	dataset_name,	image_name,	gt_name,	ilesion,	GG,	PIRADS,	Zone,	DSC Lesion,	DSC Volume,	HD95,	lesion_vol,	predicted _vol,	lesion_adc,	predicted_adc,	total false positives

    return DSC, FP, hd95, gt_vol, pred_vol,gt_adc, pred_adc, labels
        
def print_metrics(DSC, hd95, dice_3D_temp, gt_vol, pred_vol):
    # Helper to format values (handle list or scalar)
    def format_val(val):
        if isinstance(val, (list, np.ndarray)):
            return ', '.join(f'{v:.3f}' for v in val)
        else:
            return f'{val:.3f}'

    print(f'  Lesion DSC: {format_val(DSC)}')
    # print(f'  Lesion HD95: {format_val(hd95)} mm')
    print(f'  Volumetric DSC: {format_val(dice_3D_temp)}')
    print(f'  Lesion vol: {format_val(gt_vol)} mL')
    # print(f'  Prediction vol: {format_val(pred_vol)} mL')
    
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



def compute_metrics_per_lesion(labeled_gtv, labeled_pred,adc, spacing_mm):
    labels_gt = np.unique(labeled_gtv)
    labels_gt = labels_gt[labels_gt != 0]  # skip background

    DSC_list = []
    FD = 0.0
    hd95_list = []
    gt_vol_list = []
    pred_vol_list = []
    label_list=[]
    gt_adc_list=[]
    pred_adc_list=[]

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
        
        gt_adc = np.median(adc[gt_mask])
        pred_adc = np.median(adc[pred_mask])
        
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
        label_list.append(label_gt)
        gt_adc_list.append(gt_adc)
        pred_adc_list.append(pred_adc)
    
    FD= np.size(np.unique(labeled_pred[labeled_pred != 0]))-np.sum(np.array(DSC_list)>0.1)
    # print(np.size(np.unique(labeled_pred[labeled_pred != 0])))
    # print(np.sum(np.array(DSC_list)>0.1))
    return DSC_list, FD, hd95_list, gt_vol_list, pred_vol_list, gt_adc_list,pred_adc_list, label_list


#add in lesion designations:
def lesion_eval(seg,gtv,adc, spacing_mm, connected_component_analysis=True):
  #filter out small unconnected segs
   
 
    structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
    seg=np.squeeze(seg)
    
    labeled_gtv=np.squeeze(gtv)
    if connected_component_analysis: #creates lesion designations using connected component analysis, otherwise uses input labels
        labeled_gtv[labeled_gtv>0.0]=1
        labeled_gtv.astype(int)
        labeled_gtv, ncomponents_gtv = label(gtv, structure)
    
    
    
    #get rid of errant reference segmentations
    labels_gt = np.unique(labeled_gtv)
    labels_gt = labels_gt[labels_gt != 0]  # skip background
    
    kept_labels=[]
    #biggest_lesion_vol=0.0
    for label_gt in labels_gt:
        gt_mask = (labeled_gtv == label_gt)
        lesion_vol = np.sum(gt_mask) * np.prod(spacing_mm) / 1000.0
        if lesion_vol<0.05:
            labeled_gtv[labeled_gtv == label_gt]=0.0
        else:
            kept_labels.append(label_gt)
            
    ilabel=0
    for label_gt in kept_labels:
         ilabel+=1
         labeled_gtv[labeled_gtv == label_gt]=ilabel
    #labeled_gtv=np.squeeze(gtv) #logic for assumed single lesion
    ncomponents_gtv=ilabel
    assert(ilabel>0, 'no lesion in reference')
        
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
    
    DSC, FD, hd95, gt_vol, pred_vol,gt_adc, pred_adc, labels=compute_metrics_per_lesion(labeled_gtv, labeled_seg,adc, spacing_mm)
    
    #print(labels_seg)
    print('  lesions/preds, FP: %i/%i, %i' % (ncomponents_gtv,np.size(np.unique(labeled_seg[labeled_seg != 0])),FD))
    return  DSC, FD, hd95, gt_vol, pred_vol,gt_adc, pred_adc,labels  
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
        image.data=adc_norm(image.data)/2.0*3.5+1.0
        multiplier=1
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

resultspaths= [r'/lila/data/deasy/Josiah_data/Prostate/results/NKI_MSK' , r'/gpfs/home1/rbosschaert/NKI_MSK']
resultsdir = next((path for path in resultspaths if os.path.exists(path)), None)  
  
dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(root_dir,opt.name,'ct_seg_val_loss.csv')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)


dest_path= os.path.join(root_dir,opt.name) 
wt_path= os.path.join(resultsdir,'seg_test_DSC_%s_%s_%s.csv' %(opt.model_to_test,opt.name,opt.test_case))
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
# fd_results = open(wt_path, 'w')
# fd_results.write('Filename, LesionN, Lesion DSC, whole volume DSC, hd95 (mm), gt vol calculated (mL), pred vol(mL) \n')

fd_results = open(wt_path, 'w')
# fd_results.write(
#     'Model,Dataset,Filename,Lesion Filename,LesionN,Lesion DSC,Whole Volume DSC,hd95 (mm),'
#     'GT Vol Calculated (mL),Pred Vol (mL),GT ADC,Pred ADC,False Positives'
#     'Gleason Grade Group,PIRADS,Zone,ROI Volume (cc)\n'
# )
fd_results.write(
    "Model,Dataset,Filename,Lesion Filename,LesionN,Lesion DSC,Whole Volume DSC,hd95 (mm),"
    "GT Vol Calculated (mL),Pred Vol (mL),GT ADC,Pred ADC,False Positives,"
    "Gleason_Grade_Group,PIRADS,ISUP,Zone,Clinically_significant,"
    "ROI_Volume,Scanner,Voxel_size\n"
)
#fd_results.write(img_name + ',' + str(DSC) +','+ str(dice_3D_temp) + ',' + str(hd95) +'\n' )
      
seg_path=os.path.join(root_dir,opt.name,'test_segs')
if not os.path.exists(seg_path):
    os.makedirs(seg_path)    

datapaths = [r'/lila/data/deasy/Josiah_data/Prostate/nii_data', r'/gpfs/home1/rbosschaert/']
datadir = next((path for path in datapaths if os.path.exists(path)), None)  

csv_path=None
custom_names = ["img", "seg", "t2w","prost"]

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
    valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','ADC')
    #valpath_masks = os.path.join(datadir,'MR_ProstateX','Masks','T2w_GS')
    valpath_prostate_masks = os.path.join(datadir,'MR_ProstateX','Masks','Prostate')
    
    impath=valpath_adcpath
    
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files = find_common_files(valfolders, custom_names,pattern=r'.*ProstateX-(\d+).*\.nii\.gz')   
    csv_path=os.path.join(datadir,'MR_ProstateX','ADC_lesions.csv')
    
elif opt.test_case.lower()=='msk':  
    valpath_adcpath = os.path.join(datadir,'LINAC95_B_nii','Images','ADC')
    valpath_t2path = os.path.join(datadir,'LINAC95_B_nii','Images','T2w')
    valpath_masks = os.path.join(datadir,'LINAC95_B_nii','Masks','DIL')
    valpath_prostate_masks = os.path.join(datadir,'LINAC95_B_nii','Masks','Prostate')
    
    impath=valpath_adcpath
    
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]  # Replace with actual paths
    custom_names = ["img", "seg", "t2w","prost"]
    
    val_files = find_common_files(valfolders, custom_names,pattern=r'.*_P(\d+)_S(\d+).*\.nii\.gz')
    csv_path=os.path.join(datadir,'LINAC95_combined_nii','ADC_pt_csv.csv')
elif opt.test_case.lower()=='bl_sim':  
    valpath_adcpath = os.path.join(datadir,'MR_SIM34_BLfollowup_NII','Images','ADC')
    valpath_t2path = os.path.join(datadir,'MR_SIM34_BLfollowup_NII','Images','T2w')
    valpath_masks = os.path.join(datadir,'MR_SIM34_BLfollowup_NII','Masks','DIL')
    #trainpath3_masks = os.path.join(datadir,'MR_SIM20_DIRVAR_NII','Masks','DIL_T2w')
    valpath_prostate_masks = os.path.join(datadir,'MR_SIM34_BLfollowup_NII','Masks','Prostate_JJS')
    valfolders = [valpath_adcpath, valpath_masks,valpath_t2path,valpath_prostate_masks]
    val_files = find_common_files(valfolders, custom_names,pattern=r'.*_P(\d+).*\.nii\.gz$')  



adc_norm = ScaleIntensityRangePercentiles(
    lower=0.0,  # 0th percentile
    upper=99.0, # 99th percentile
    b_min=-1.0, # target min
    b_max=1.0,  # target max
    clip=True   # clip values outside percentiles
)    

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
        CropForegroundd(keys=["img","seg","prost", "t2w"], source_key= "prost", margin=[64,64,opt.extra_neg_slices + (opt.nslices - 1) / 2]),
        
        
        #RandSpatialCropd(keys=["img","t2w","seg"], roi_size=(128, 128, opt.nslices),random_size=False),,
        CenterSpatialCropd(keys=["img","t2w", "prost","seg"], roi_size=[128, 128,-1]),
        EnsureTyped(keys=["img", "seg","prost", "t2w"]),
        AsDiscreted(keys=["prost","seg"], rounding="torchrounding"),
    ]
  )

post_transforms = Compose([
        EnsureTyped(keys=["pred","seg","prost","img"]),
        Invertd(
            keys=["pred","seg","prost","img"],
            transform=val_transforms,
            orig_keys="img",
            meta_keys=["pred_meta_dict","pred_meta_dict","pred_meta_dict","image_meta_dict"],
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys=["pred","seg","prost","img"], argmax=False),

    ])


val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)

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
        
        print('******************************************************')

        adc=torch.squeeze(adc,0)
        if adc.numel() != 0:
            predictions=[]
            with torch.no_grad():
                with autocast(enabled=True):
                    for model in models:
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
                        predictions.append(pred)
                        
            val_data["pred"] = torch.mean(torch.stack(predictions), dim=0)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            seg = from_engine(["pred"])(val_data)[0]
            adc_val = from_engine(["img"])(val_data)[0]              
            gtv = from_engine(["seg"])(val_data)[0]
            prostate = from_engine(["prost"])(val_data)[0]
            
            
        
            
            gtv = np.array(gtv)
            gtv=np.squeeze(gtv)
            adc_val=3500.0*(adc_val+1.0)/2.0 #placeholder estimate, NOT CORRECT!
            adc_val = np.array(adc_val)
            adc_val=np.squeeze(adc_val)
            
            if  np.sum(gtv)>5:
                    
                prostate = np.array(prostate)
                prostate=np.squeeze(prostate)
                prostate[prostate>0.5]=1.0
                prostate[prostate<1.0]=0.0
                
                prostate=ndimage.binary_dilation(prostate).astype(prostate.dtype)
                if np.sum(prostate*gtv)<(0.9*np.sum(gtv)):
                    print('prostate mask issue')
                    prostate[prostate<1.0]=1.0
                    
                seg = np.array(seg)
                seg=np.squeeze(seg)
                
                seg[seg >= opt.seg_threshold]=1.0
                seg[seg < opt.seg_threshold]=0.0
                
                seg_filtered= np.array(seg)
                seg_filtered[prostate < 1.0]=0.0
                
                # #filter out small unconnected segs @0.5x0.5x3 27 vox ~ 2 mL
                # structure = np.ones((3, 3, 3), dtype=int)  # this defines the connection filter
                # labeled, ncomponents = label(seg_filtered, structure)    
                # for ilabel in range(0,ncomponents+1):
                #     if np.sum(labeled==ilabel+1)<15:
                #         seg_filtered[labeled==ilabel]=0
        
                img_name=adc.meta['filename_or_obj'][0].split('/')[-1]
                gtv_name=label_val.meta['filename_or_obj'][0].split('/')[-1]
                
                impath=os.path.dirname(adc.meta['filename_or_obj'][0])
                cur_rd_path=os.path.join(impath,img_name)
                im_obj = sitk.ReadImage(cur_rd_path)
                spacing_mm=im_obj.GetSpacing()
                seg_temp=np.array(seg_filtered)
                gtv=np.squeeze(gtv)

                seg_flt=seg_temp.flatten()
                gt_flt=gtv.flatten()
                gt_flt=gt_flt > 0.5
                
                intersection = np.sum(seg_flt * (gt_flt > 0))
                dice_3D_temp=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt > 0) + smooth)
                dice_3D.append(dice_3D_temp)
                
                DSC, FP, hd95, gt_vol, pred_vol,gt_adc, pred_adc, labels=process_with_csv(img_name,gtv_name,opt.name,opt.test_case, seg_filtered, gtv, adc_val, spacing_mm, fd_results, csv_path)
                
                seg = np.transpose(seg, (2, 1, 0))
                seg_filtered = np.transpose(seg_filtered, (2, 1, 0))
                prostate = np.transpose(prostate, (2, 1, 0))
                
                seg = sitk.GetImageFromArray(seg)
                seg = copy_info(im_obj, seg)
                sitk.WriteImage(seg,  os.path.join(seg_path,'seg_%s' % img_name))
                
                
                seg_filtered = sitk.GetImageFromArray(seg_filtered)
                seg_filtered = copy_info(im_obj, seg_filtered)
                sitk.WriteImage(seg_filtered,  os.path.join(seg_path,'filteredseg_%s' % img_name))
                
                prostate = sitk.GetImageFromArray(prostate)
                prostate = copy_info(im_obj, prostate)
                sitk.WriteImage(prostate,  os.path.join(seg_path,'prostateseg_%s' % img_name))

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
        
    recall=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice))
    print('recall: %f' % recall)
    
    precision=np.sum(np.array(Lesion_Dice)>0.1).astype(float)/float(np.size(Lesion_Dice)+np.sum(False_positives))
    
    # print(precision)
    print('precision: %f' % precision)
    fd_results.flush()  
    
    wt2_path= os.path.join(resultsdir,'Result_summary_%s_%s_%s.csv' %(opt.model_to_test,opt.name,opt.test_case))
    fd_results_sum = open(wt2_path, 'w')
    fd_results_sum.write('name, median Lesion DSC, median whole volume DSC, median lesion hd95 (mm), precision, recall \n')
    fd_results_sum.write(opt.name + ',' + str(median_Lesion_Dice) +','+ str(np.nanmedian(dice_3D)) + ',' + str(median_hd95) + ',' + str(precision) + ',' + str(recall) +'\n')
    fd_results_sum.flush()  
