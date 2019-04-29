import SimpleITK as sitk
from glob import glob
import subprocess
import numpy as np
import os
from skimage import io
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

# -*- coding: utf-8 -*-

def apply_n4(path):
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c*/*.mha')
    n4_bfc(t1[0])
    n4_bfc(t1c[0])


def n4_bfc(input_path):
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = input_path
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.output_image = input_path.replace('.mha', '_n4.mha')
    n4.run()

'''
def n4_correction(im_input):
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = im_input
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.output_image = im_input.replace('.nii.gz', '_corrected.nii.gz')
    n4.run()

'''
# updating to look for n4 suffixed scans
def load_scans(path):
    """
    scans = [flair, t1, t1c, t2, gt]
    output shape: (155, 5, 240, 240)
    """
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c.*/*.mha')
    t2 = glob(path + '/*T2*/*.mha')
    gt = glob(path + '/*OT*/*.mha')

    mod_paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    mods = []
    for mod_path in mod_paths:
        mod_img = io.imread(mod_path, plugin='simpleitk').astype('float')
        mod_array = np.array(mod_img)
        mods.append(mod_array)

    data_mods = np.array(mods)
    data_slices = np.zeros((155, 5, 240, 240))
    for i in range(155):
        for j in range(5):
            data_slices[i,j,:,:] = data_mods[j][i,:,:]

    return data_slices


def normalize(scans):
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
    b, t = np.percentile(slice, (0.5,99.5))
    slice = np.clip(slice, b, t)
    if np.std(slice) == 0:
        return slice
    else:
        return (slice - np.mean(slice)) / np.std(slice)
    
