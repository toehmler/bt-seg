# @author: Trey Oehmler

from PIL import Image
from keras.models import load_model
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import configparser
import sys    

'''
==================== predict.py ==================== 
Makes a prediction using a given model, patient no and slice no

Input:  (1) name of model to use (eg. m1_5)
        (2) patient number       
        (3) slice number 


Output: (1) .npy file representing prediction as numpy array 
        (2) img of prediction with mri and gt 
        (3) saves text file with metrics (dice score, etc)

Shape:  [Out] (43264,)

Usage:  predict.py [model_name] [patient_no] [slice_no]
==================================================== 

TODO
- use test.py as model
'''

config = configparser.ConfigParser()
root_path = config['paths']['processed']
path = root_path

model_name = sys.srgv[1]
patient_no = sys.argv[2]
slice_no = sys.argv[3]

data_img = Image.open(config.train_root + 'data/test/pat210_60_data.png')
test_data = np.asarray(data_img)
test_data = test_data.reshape(4, 240, 240)

data = np.zeros((240, 240, 4))
for i in range(4):
    data[:,:,i] = test_data[i,:,:]

test_patches = extract_patches_2d(data, (33,33))

model = load_model('/home/trey/bt-seg/Outputs/Models/Trained/m1_3.h5')

prediction = model.predict_classes(test_patches)
np.save('/home/trey/bt-seg/Outputs/210_60_m3.npy', prediction)








