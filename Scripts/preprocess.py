from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]
out_path = config['processed']

for patient_no, path in enumerate(paths):
    print("Processing patient " + str(patient_no))
    scans = patient.load_scans(path)
    normed_scans = patient.normalize(scans) # (155, 240, 240, 5)
    for slice_no, slice in enumerate(normed_scans):

        strip = slice[:,:,:4]
        label = slice[:,:,4]

        if np.max(strip) != 0: # set values < 1
            strip /= np.max(strip)
        if np.min(strip) <= -1: # set values > -1
            strip /= abs(np.min(strip))

        out_strip = np.zeros((240,240,5))
        out_strip[:,:,4] = label
        out_strip[:,:,:4] = strip
        if patient_no < 190:
            np.save(out_path + '/train/pat{}_{}.npy'.format(patient_no, slice_no), out_strip)
        else:
            np.save(out_path + '/test/pat{}_{}.npy'.format(patient_no, slice_no), out_strip)
    break        

