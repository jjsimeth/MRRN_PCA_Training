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
import datetime
import SimpleITK as sitk
import torch.multiprocessing

import torch.utils.data
from options.train_options import TrainingOptions
from models.models import create_model
# from util import util
from PCA_DIL_inference_utils import sliding_window_inference 
import pathlib


# from pathlib import Path
from glob import glob

from typing import Tuple, Optional
import monai
from monai.handlers.utils import from_engine
from monai.data import DataLoader, create_test_image_3d, MetaTensor
from monai.data import decollate_batch

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


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




def scale_ADC(image: MetaTensor):
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


def visualize_metatensor_slices(adc: "MetaTensor", t2: "MetaTensor", label_data: "MetaTensor",
                                prediction_data: np.ndarray, save_folder: Optional[pathlib.Path] = None,
                                dice_score:Optional[float] = None, epoch:Optional[int] = None,
                                input_data:Optional[str] = None) -> Optional[
    pathlib.Path]:
    """
    Visualize all slices from ADC and T2 MetaTensor objects with label contours and prediction overlays in a single figure.

    This function takes ADC, T2, label MetaTensors and a prediction NumPy array, and visualizes
    all slices in one figure. The resulting plot shows ADC and T2 images in two columns,
    with label contours and prediction contours overlaid on both images.

    Args:
        adc (MetaTensor): A 5D MetaTensor object containing the ADC image data.
        t2 (MetaTensor): A 5D MetaTensor object containing the T2 image data.
        label_data (MetaTensor): A 5D MetaTensor object containing the label contour data.
        prediction_data (np.ndarray): A 3D NumPy array containing the prediction contour data.
        save_folder (Path, optional): Path to the folder where the visualization should be saved.
                                      If None, the plot is displayed but not saved.

    Returns:
        Optional[Path]: The path to the saved visualization if save_folder is provided, else None.
    """
    # Ensure MetaTensor scan inputs are 5D tensors
    if any(tensor.ndim != 5 for tensor in [adc, t2, label_data]):
        raise ValueError("All scan MetaTensor inputs must have 5 dimensions")

    # Select appropriate dimensions and move to CPU
    adc = adc[0, 0, :, :, :].cpu().numpy()
    t2 = t2[0, 0, :, :, :].cpu().numpy()
    label_data = label_data[0, 0, :, :, :].cpu().numpy()

    # Ensure prediction_data is a 3D NumPy array
    if prediction_data.ndim != 3:
        raise ValueError("prediction_data must be a 3D NumPy array")

    num_slices = adc.shape[2]  # Assuming the last dimension is the number of slices

    # Create figure with two columns
    fig, axes = plt.subplots(num_slices, 2, figsize=(10, 5 * num_slices))

    def create_suptitle(dice_score=None, epoch=None, input_data=None):
        title = "ADC and T2 Slices with Ground Truth and Prediction Contours"

        if dice_score is not None:
            title += f"\nDice: {dice_score}"

        additional_info = []
        if epoch is not None:
            additional_info.append(f"Epoch: {epoch}")
        if input_data is not None:
            l_input_data= input_data.split(',')
            for line in l_input_data:
                additional_info.append(f"Input: {line}")
        if additional_info:
            title += f"\n{' | '.join(additional_info)}"

        return title

    fig.suptitle(create_suptitle(dice_score=dice_score, epoch=epoch, input_data=input_data), fontsize=16)

    for slice_index in range(num_slices):
        # Process images for the current slice
        adc_slice = adc[:, :, slice_index]
        t2_slice = t2[:, :, slice_index]

        # Indicate plotted position
        ax_adc, ax_t2 = axes[slice_index, :]

        # Process contour data
        label_slice = label_data[:, :, slice_index]
        prediction_slice = prediction_data[:, :, slice_index]

        # Plot ADC slice
        im_adc = ax_adc.imshow(adc_slice, cmap='gray')
        ax_adc.contour(label_slice, colors='r', linewidths=0.5)
        ax_adc.contour(prediction_slice, colors='b', linewidths=0.5)
        ax_adc.axis('off')
        ax_adc.set_title(f'ADC Slice {slice_index}')

        # Plot T2 slice
        im_t2 = ax_t2.imshow(t2_slice, cmap='gray')
        ax_t2.contour(label_slice, colors='r', linewidths=0.5)
        ax_t2.contour(prediction_slice, colors='b', linewidths=0.5)
        ax_t2.axis('off')
        ax_t2.set_title(f'T2 Slice {slice_index}')

    # Create legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Ground Truth'),
        Patch(facecolor='blue', edgecolor='blue', label='Prediction')
    ]

    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95)  # Make room for the legend and title

    # Save the visualization if save_folder is provided
    if save_folder:
        save_folder = pathlib.Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%H_%M_%S_%f")
        file_name = f'train_vis_all_slices_{current_time}.png'
        image_path = save_folder / file_name
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to: {image_path}")
        return image_path
    else:
        plt.show()
        return None

def train_stuff():

    opt = TrainingOptions().parse()
    opt.isTrain=True

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

    #unfreeze final layer and Channel input block for finetuning
    model.netSeg_A.CNN_block1.requires_grad_(True)
    model.netSeg_A.out_conv.requires_grad_(True)
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


    #get nii and seg data prosx
    #nmodalities=2
    # impath = os.path.join(root_dir,'training_data') #load data
    # images = sorted(glob(os.path.join(root_dir, "adc*.nii.gz")))
    # images = sorted(glob(os.path.join(impath, "*_ep2d_diff_*.nii.gz"))) #ivim adc keywords from filename
    #segs = sorted(glob(os.path.join(impath, "*_ADC_ROI*.nii.gz")))
    # segs = sorted(glob(os.path.join(impath, "*t2_tse_tra*_ROI.nii.gz")))
    # images_t2w = sorted(glob(os.path.join(root_dir, "t2w*.nii.gz")))
    # images_t2w = sorted(glob(os.path.join(impath, "*_t2_tse*.nii.gz"))) #t2w keywords from filename

    # NKI data
    impath = os.path.join(root_dir,'training_data\\nki_resampled') #load data
    images = sorted(glob(os.path.join(impath, "*adc*.nii")))
    segs = sorted(glob(os.path.join(impath, "*LES*.nii")))
    images_t2w = sorted(glob(os.path.join(impath, "*tt2*.nii"))) #t2w keywords from filename

    # print(images)
    # print(segs)
    # print(images_t2w)
    #can add additional modalities with thier keyname
    #images_ktrans = sorted(glob(os.path.join(root_dir, "ktrans*.nii.gz")))

    # total_dataset= 10
    total_dataset= 156
    n_val=round(total_dataset*0.3)
    n_train=total_dataset-n_val

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

    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)


    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    check_data = monai.utils.misc.first(train_loader)
    print("first patch's shape: ", check_data["img"].shape, check_data["seg"].shape, check_data["t2w"].shape)

    epoch_loss_values = []
    start_training = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    num_epochs = 10000
    best_dice=0
    selected_slice = 2
    dice_over_time = {}
    for epoch in range(num_epochs):
        # if epoch==100:
        #     model.netSeg_A.out_conv.requires_grad_(True)
        if epoch==500:
            model.netSeg_A.CNN_block2.requires_grad_(True)
            model.netSeg_A.RU11.requires_grad_(True)

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

            if torch.sum(labels)>0:
                adc=scale_ADC(adc)

                inputs=torch.cat((adc,t2w),dim=1)
                labels=labels[:,selected_slice,:,:]



                inputs, labels = mixup(inputs, labels, np.random.beta(1.0, 1.0))
                #inputs, labels = mixup(inputs, labels, np.random.beta(0.2, 0.2))
                labels=  torch.clamp(labels,0.001,0.999) #label smoothing

                model.set_input_sep(inputs,labels)
                model.optimize_parameters()
        if (epoch%50)==0:
                model.get_curr_lr()

        if (epoch%5)==0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}. Current datetime: {datetime.datetime.now()}")
            print("-" * 10)
            with torch.no_grad(): # no grade calculation
                loss_list = []
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
                    print(f"dice {dice_2D_temp} for data: {val_loader.dataset.data[step-1]}")
                    if (epoch%50)==0:
                        # visualize for inspection
                        visualization_folder = pathlib.Path(
                            r"D:\Projects\PCA_Segmentation_MRRN_training\MRRN_PCA_Training\visualisation\mixup\nki_resampled")
                        visualize_metatensor_slices(adc=adc, t2=t2w, label_data=label_val,
                                                    prediction_data=seg_temp, save_folder=visualization_folder,
                                                    dice_score=dice_2D_temp, epoch=epoch, input_data=str(val_loader.dataset.data[step-1]))
                    else:
                        pass
                    dice_2D.append(dice_2D_temp)
                    val_inp1 = val_inputs[0, :, :, :, :]
                    val_inp2 = val_inputs.squeeze(dim=0)
                    label_inp1 = label_val[0, 0, :, :, :]
                    label_inp2 = label_val.squeeze(dim=0)
                    # seg_loss = model.cal_seg_loss(model.netSeg_A, val_inp2, label_inp2)
                    # loss_list.append(seg_loss.item())




                # average_loss = np.average(loss_list)
                dice_2D=np.average(dice_2D)

                dice_over_time[f'{epoch}'] = [dice_2D,datetime.datetime.now()]
                dice_df = pd.DataFrame(dice_over_time, index=['dice','datetime'])
                dice_df.to_excel(f"dice_over_time_{start_training}.xlsx")
                print(f'Updated excel at :"dice_over_time_{start_training}.xlsx"')
                print('epoch %i' % epoch, 'DSC  %.2f' % dice_2D, ' (best: %.2f)'  % best_dice)
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

                            vol_data = [post_transforms(i) for i in decollate_batch(vol_data)]
                            seg_out= from_engine(["pred"])(vol_data)[0]
                            seg_out = np.array(seg_out)
                            seg_out=np.squeeze(seg_out)
                            seg_out[seg_out >= 0.5]=1.0
                            seg_out[seg_out < 0.5]=0.0
                            seg_out = np.transpose(seg_out, (2, 1, 0))


                            cur_rd_path=os.path.join(impath,img_name)
                            im_obj = sitk.ReadImage(cur_rd_path)
                            seg_out = sitk.GetImageFromArray(seg_out)
                            seg_out = copy_info(im_obj, seg_out)
                            sitk.WriteImage(seg_out, os.path.join(os.path.dirname(img_name),
                                                                  'seg_%s' % os.path.basename(img_name)))

                        model.save('AVG_best_finetuned')
                        best_dice = dice_2D