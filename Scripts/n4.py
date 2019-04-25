# Author: Trey Oehmler
from Utils import patient
import configparser
import os
import json
'''
==================== n4.py ====================  
Applies n4 bias field correction to t1 and t1c

Input:  none

Output: (1) t1_n4.mha and t1c_n4.mha 
            (for every patient)

Usage:  n4.py
===============================================
'''

# load and parse config file

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]

for patient_no, path in enumerate(paths):
    print('Applying n4 bias field correction on patient: ' + str(patient_no))
    print('Path: '+ + str(path))
    patient.apply_n4(path)
    break





