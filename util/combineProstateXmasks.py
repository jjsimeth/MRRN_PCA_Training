# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:45:34 2025

@author: SimethJ
"""

import os
import re
import numpy as np
import nibabel as nib
from collections import defaultdict

def load_nii(file_path):
    """Load a NIfTI file and return the image data and affine matrix."""
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def save_nii(data, affine, output_path):
    """Save the combined binary image as a NIfTI file."""
    combined_nii = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(combined_nii, output_path)

def combine_findings(src_folder, dest_folder):
    """Load all matching files per ID, combine them, and save the result."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Dictionary to store files grouped by ID
    findings_dict = defaultdict(list)
    
    # Regular expression to extract the ID (e.g., ProstateX-0001)
    pattern = re.compile(r'^(ProstateX-\d+)-Finding\d+.*\.nii\.gz$')
    
    # Group files by patient ID
    for file in os.listdir(src_folder):
        match = pattern.match(file)
        if match:
            patient_id = match.group(1)
            findings_dict[patient_id].append(os.path.join(src_folder, file))
    
    # Process each patient ID
    for patient_id, file_list in findings_dict.items():
        combined_data = None
        affine = None
        
        for file_path in file_list:
            data, affine = load_nii(file_path)
            
            if combined_data is None:
                combined_data = np.zeros_like(data, dtype=np.uint8)
            
            combined_data = np.logical_or(combined_data, data).astype(np.uint8)
        
        # Save combined binary mask
        output_file = os.path.join(dest_folder, f"{patient_id}_all_findings.nii.gz")
        save_nii(combined_data, affine, output_file)
        print(f"Saved: {output_file}")
if __name__ == "__main__":
    src_folder = r"Y:\\Prostate DIL segmentation\\Data\\Main_Datasets\\Open_Source\\PROSTATEx_masks-master\\Files\\lesions\\Masks\\ADC"
    dest_folder = r"Y:\\Prostate DIL segmentation\\Data\\Main_Datasets\\Open_Source\\PROSTATEx_masks-master\\Files\\lesions\\Combined_Masks\\ADC"
    combine_findings(src_folder, dest_folder)
    src_folder = r"Y:\\Prostate DIL segmentation\\Data\\Main_Datasets\\Open_Source\\PROSTATEx_masks-master\\Files\\lesions\\Masks\\T2"
    dest_folder = r"Y:\\Prostate DIL segmentation\\Data\\Main_Datasets\\Open_Source\\PROSTATEx_masks-master\\Files\\lesions\\Combined_Masks\\T2"
    combine_findings(src_folder, dest_folder)