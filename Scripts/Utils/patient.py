# Author: Trey Oehmler

import SimpleITK as sitk
from glob import glob
import subprocess
import numpy as np
import os
from skimage import io
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from tqdm import tqdm 
'''
==================================================
NOT USING
--------------------------------------------------

def apply_n4(path):
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c*/*.mha')
    n4_bfc(t1[0])
    n4_bfc(t1c[0])

def n4_bfc(input_path):
    print("Applying bias correction...")
    print("Input: {}".format(input_path))
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = input_path
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.output_image = input_path.replace('.mha', '_n4.mha')
    n4.run()
==================================================
'''

def load_scans(path):
    '''
    output shape: (5, 155, 240, 240)
    '''
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c.*/*.mha')
    t2 = glob(path + '/*T2*/*.mha')
    gt = glob(path + '/*OT*/*.mha')

    mod_paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    mods = []
    for i in tqdm(range(5)):
        mod_path = mod_paths[i]
        mod_img = io.imread(mod_path, plugin='simpleitk').astype(np.float32)
        mod_array = np.array(mod_img)
        mods.append(mod_array)

    data_mods = np.array(mods)
    return data_mods


def normalize(scans):
    '''
    output shape: (155, 240, 240, 5)
    '''
    normed_scans = np.zeros((155, 240, 240, 5))
    # exclude ground truth
    normed_scans[:,:,:,4] = scans[4] 
    for i in range(4):
        norm_mod = normalize_mod(scans[i])
        normed_scans[:,:,:,i] = norm_mod
    return normed_scans


def normalize_mod(mod):
    x_start = mod.shape[0] // 4
    x_range = mod.shape[0] // 2
    y_start = mod.shape[1] // 4
    y_range = mod.shape[1] // 2
    z_start = mod.shape[2] // 4
    z_range = mod.shape[2] // 2
    roi = mod[x_start:x_start+x_range, y_start:y_start+y_range, z_start:z_start+z_range]
    return (mod - np.min(roi)) / np.std(roi)
   
