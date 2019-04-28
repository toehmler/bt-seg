from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import scipy.misc
from PIL import Image


with open('config.json') as config_file:
    config = json.load(config_file)

root = config['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]
out_path = config['processed']

for patient_no, path in enumerate(paths):
    print("Processing patient " + str(patient_no))
    scans = patient.load_scans(path)
    normed_scans = patient.normalize(scans) # (155, 5, 240, 240)
    for slice_no, slice in enumerate(normed_scans):
        strip = slice[:4,:,:]
        label = slice[4,:,:]
        
        if np.max(strip) != 0:
            strip /= np.max(strip)

        if np.min(strip) <= -1:
            strip /= abs(np.min(strip))
        
        strip = strip.reshape(960,240)
        label = label.astype(np.uint8)
        if patient_no < 190:
            imageio.imwrite(out_path+'/train/pat{}_{}_data.png'.format(patient_no, slice_no), strip)
            imageio.imwrite(out_path+'/train/pat{}_{}_label.png'.format(patient_no, slice_no), label)
        else:
            imageio.imwrite(out_path+'/test/pat{}_{}_data.png'.format(patient_no, slice_no), strip)
            imageio.imwrite(out_path+'/test/pat{}_{}_label.png'.format(patient_no, slice_no), label)
