from Utils import config
from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import configparser

config = configparser.ConfigParser()
root = config['paths']['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]
out_path = config['paths']['processed']

for patient_no, path in enumerate(paths):
    print("Processing patient " + str(patient_no))
    scans = patient.load_scans(path)
    normed_scans = patient.normalize(scans)

    patient_data = normed_scans[:,:4,:,:]
    patient_labels = normed_scans[:,4,:,:]

    for slice_no, data_slice in enumerate(patient_data):
        strip = data_slice.reshape(960,240)
        if np.max(strip) != 0: # set values < 1
            strip /= (np.max(strip)        
        if np.min(strip) <= -1: # set values > -1
            strip /= abs(np.min(strip))             
            
        
        if patient_no < 190:
            np.save(out_path + '/data/train/pat{}_{}_data.npy'.format(patient_no, slice_no), strip)
        else:
            np.save(out_path + '/data/test/pat{}_{}_data.npy'.format(patient_no, slice_no), strip)


    for slice_no, label_slice in enumerate(patient_labels):

        if patient_no < 190:
            np.save(out_path + '/labels/train/pat{}_{}_label.npy'.format(patient_no, slice_no), label_slice
            imageio.imwrite(out_path + '/labels/train/pat{}_{}_label.png'.format(patient_no, slice_no), label_img)
        else:
            imageio.imwrite(out_path + '/labels/test/pat{}_{}_label.png'.format(patient_no, slice_no), label_img)



