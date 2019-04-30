# Author: Trey Oehmler
from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import scipy.misc
from PIL import Image
from skimage import io

'''
==================================================
preprocess.py 
--------------------------------------------------
(1) Loads scans each patient (.mha files)
(2) Normalizes pixel itensities 
(3) Splits patients into training and test sets
(4) Saves np.array (float32) for each patient
==================================================

'''
with open('config.json') as config_file:
    config = json.load(config_file)

root = config['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]
out_path = config['processed']


for i in tqdm(range(len(paths))):
    scans = patient.load_scans(paths[i]) # (5, 155, 240, 240)
    patient_data = patient.normalize(scans) # (155, 240, 240, 5)
    # split patients into training and testing sets
    if i < 190:
        train_data = np.memmap(out_path + '/train/pat_{}.dat'.format(i), dtype='float32', mode='w+', shape=(155,240,240,5))
        train_data[:,:,:,:] = patient_data[:,:,:,:]
        del train_data
    else:
        test_data = np.memmap(out_path + '/test/pat_{}.dat'.format(i), dtype='float32', mode='w+', shape=(155,240,240,5))
        test_data[:,:,:,:] = patient_data[:,:,:,:]
        del test_data
    break
