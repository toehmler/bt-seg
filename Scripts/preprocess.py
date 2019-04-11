from Utils import config
from Utils import data
from Utils import patient
import imageio
import numpy as np
import matplotlib.pyplot as plt

paths =  data.patient_paths(config.root)

# apply n4 bias correction to t1 and t1c scans
for i, path in enumerate(paths):
    print('Applying n4 bias field correction for patient ' + str(i))
    patient.apply_n4(path)
    scans = patient.load_scans(path)
    norm_scans = patient.normalize(scans)
    patient.save_strips(norm_scans, i)
    print('Finished processing patient ' + str(i))

