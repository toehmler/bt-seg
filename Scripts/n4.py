# Author: Trey Oehmler
from Utils import patient
import configparser
'''
==================== n4.py ====================  
Applies n4 bias field correction to t1 and t1c

Input:  none

Output: (1) t1_n4.mha and t1c_n4.mha 
            (for every patient)

Usage:  n4.py
===============================================
'''

config = configparser.ConfigParser()
config.read('config.ini')
root = config['DEFAULT']['brats']
paths = os.listdir(root)
paths = [os.path.join(root, name) for name in paths if 'pat' in name.lower()]

for patient_no, path in enumerate(paths):
    print('Applying n4 bias field correction on patient: ' + patient_no)
    patient.apply_n4(path)
    break






