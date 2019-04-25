import SimpleITK as sitk
from glob import glob
import subprocess
import numpy as np
import os
from skimage import io

# -*- coding: utf-8 -*-

def apply_n4(path):
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c*/*.mha')
    out_path = path[:-4] + '_n4.mha'
    n4_bfc(t1[0], out_path)
    n4_bfc(t1c[0], out_path)


def n4_bfc(path, out_path):
    inputImage = sitk.ReadImage(path)
    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
    inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    output = corrector.Execute(inputImage,maskImage)
    sitk.WriteImage(output, out_path)
    print("Finished N4 Bias Field Correction.....")


# updating to look for n4 suffixed scans
def load_scans(path):
    """
    scans = [flair, t1, t1c, t2, gt]
    output shape: (155, 5, 240, 240)
    """
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1_n4*/*.mha')
    t1c = glob(path + '/*T1c_n4*/*.mha')
    t2 = glob(path + '/*T2*/*.mha')
    gt = glob(path + '/*OT*/*.mha')

    mod_paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    mods = []
    for mod_path in mod_paths:
        mod_img = io.imread(mod_path, plugin='simpleitk').astype('float')
        mod_img = sitk.ReadImage(mod_path)
        mod_array = sitk.GetArrayFromImage(mod_img)
        mods.append(mod_array)

    data_mods = np.array(mods)
    data_slices = np.zeros((155, 5, 240, 240))
    for i in range(155):
        for j in range(5):
            data_slices[i,j,:,:] = data_mods[j][i,:,:]

    return data_slices


def normalize(scans):
    """
    Expecting input shape of (155, 5, 240, 240)
    """
    normed_scans = np.zeros((155, 5, 240, 240))
    
    #exclude gt from normalization
    gt = scans[:,4,:,:]

    for i in range(155):
        for j in range(4):
            normed_slice = norm_slice(scans[i][j,:,:])
            normed_scans[i,j,:,:] = normed_slice
    normed_scans[:,4,:,:] = gt
    return normed_scans

    
def norm_slice(slice):
    """
    """
    b, t = np.percentile(slice, (0.5,99.5))
    slice = np.clip(slice, b, t)
    if np.std(slice) == 0:
        return slice
    else:
        normed_slice = (slice - np.mean(slice)) / np.std(slice)
        return normed_slice



