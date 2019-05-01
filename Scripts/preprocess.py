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
import skimage
from skimage import io
import warnings

warnings.filterwarnings("ignore")

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
out = config['processed']


for i in tqdm(range(len(paths))):
    scans = patient.load_scans(paths[i]) # (5, 155, 240, 240)
    patient_data = patient.normalize(scans) # (155, 5, 240, 240)
    for j in tqdm(range(155)):
        strip = patient_data[j,:4,:,:]
        label = patient_data[j,4,:,:]
        strip = strip.reshape(960,240)
        if (np.max(strip)) != 0:
            strip /= np.max(strip)
        if (np.min(strip)) != 0:
            strip /= abs(np.min(strip))
        strip_img = skimage.img_as_uint(strip)
        if i < 190:
            io.imsave('{}/train/pat_{}_{}_strip.png'
                      .format(out, i, j), strip_img)
            io.imsave('{}/train/pat_{}_{}_label.png'
                      .format(out, i, j), label_img)
        else:
            io.imsave('{}/test/pat_{}_{}_strip.png'
                      .format(out, i, j), strip_img)
            io.imsave('{}/test/pat_{}_{}_label.png'
                      .format(out, i, j), label_img)




