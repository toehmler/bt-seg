import SimpleITK as sitk
from glob import glob
import subprocess
import numpy as np
from config import *
import os

# -*- coding: utf-8 -*-
"""patient.py 

TODO
* reshape load scans to output (155, 5, 240, 240)
* normalized slices except for ground truth
* reshape to strips (155, 1200, 240)

"""
def apply_n4(path):
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c*/*.mha')
    n_dims = 3
    n_iters = '[20,20,10,5]'
    out_path = path[:-4] + '_n4.mha'
    n4_bfc(path, str(n_dims), n_iters, out_path)


def n4_bfc(path, n_dims, n_iters, out_path):
    n4 = n4biasfieldcorrection(output_image=out_path)
    n4.inputs.dimension = n_dims
    n4.input_image = path
    n4.inputs.n_iterations = ast.literal_eval(n_iters)
    print("running n4 bias correction for " + path);
    n4.run()
    print("finished")


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


def save_strips(scans, patient_num):
    """
    Expecting input shape of (155, 5, 240, 240)
    Generates 155 x (1200, 240) stips
    """
    for slice in scans:
        strip = slice.reshape(1200, 240)
        if np.max(strip) != 0:
            strip /= np.max(strip)

        if np.min(strip) <= -1:
            strip /= abs(np.min(strip))
        strip = 255 * strip
        img = strip.astype(np.uint8)
        out_path = config.root + '/strips/pat' + str(patient_num) +'.png'
        imageio.imwrite(out_path, img)
            
    




