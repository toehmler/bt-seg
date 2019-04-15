from Utils import config
from Utils import data
from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt

paths =  data.patient_paths(config.root)
out_path = '/home/trey/data/processed'
#out_path = '/Users/treyoehmler/dev/tumors/data/processed'


for patient_no, path in enumerate(paths):
    print("Processing patient " + str(patient_no))
    scans = patient.load_scans(path)
    normed_scans = patient.normalize(scans)

    patient_data = normed_scans[:,:4,:,:]
    patient_labels = normed_scans[:,4,:,:]

    for slice_no, data_slice in enumerate(patient_data):
        strip = data_slice.reshape(960,240)
        if np.max(strip) != 0:
            strip /= (np.max(strip) - np.min(strip))
        
        if np.min(strip) != np.max(strip):
            strip -= np.min(strip)
            strip *= (255 / np.max(strip) - np.min(strip))
        
        strip_img = strip.astype(np.uint8)
        if patient_no < 190:
            imageio.imwrite(out_path + '/data/train/pat{}_{}_data.png'.format(patient_no, slice_no), strip_img)
        else:
            imageio.imwrite(out_path + '/data/test/pat{}_{}_data.png'.format(patient_no, slice_no), strip_img)



    for slice_no, label_slice in enumerate(patient_labels):
        label_img = label_slice.astype(np.uint8)

        if patient_no < 190:
            imageio.imwrite(out_path + '/labels/train/pat{}_{}_label.png'.format(patient_no, slice_no), label_img)
        else:
            imageio.imwrite(out_path + '/labels/test/pat{}_{}_label.png'.format(patient_no, slice_no), label_img)



    










'''

test = paths[140]
scans = patient.load_scans(test)
norm_scans = patient.normalize(scans)

sam = norm_scans[:,:4,:,:]
labels = norm_scans[:,4,:,:]
sample = sam[70] 
label = labels[70]
strip = sample.reshape(960, 240)

if np.max(strip) != 0:
    strip /= (np.max(strip) - np.min(strip))

if np.min(strip) != np.max(strip):
    strip -= np.min(strip)
    strip *= (255 / (np.max(strip) - np.min(strip)))

img = strip.astype(np.uint8)
imageio.imwrite('test.png', img)

print(np.min(label))
print(np.max(label))
if np.max(label) != 0:
    label /= (np.max(label) - np.min(label))

label = 255 * label

print(np.min(label))
print(np.max(label))

truth = label.astype(np.uint8)
imageio.imwrite('label_test.png', truth)

'''


'''


for i, path in enumerate(paths):
    print('Applying n4 bias field correction for patient ' + str(i))
    patient.apply_n4(path)
    scans = patient.load_scans(path)
    norm_scans = patient.normalize(scans)
    patient.save_strips(norm_scans, i, config.root)
    print('Finished processing patient ' + str(i))
'''


# apply n4 bias correction (save t1 and t1c_n4.mha files)

# normalize pizel intensities

# split data into training and testing 
# ( use same paths array returned from data.patients_paths()

# save training data

# save training data

# load + save training labels

# load + save testing labels
