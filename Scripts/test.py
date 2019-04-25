# Author: Trey Oehmler

import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.image import extract_patches_2d
'''
===============================================================
'''
import imageio
from PIL import Image
import sys
import matplotlib.pyplot as plt
import configparser

''' 
==================== test.py ============================ 
Tests a given model on a given patient

Input:  (1) name of model to test (eg. m1_5) 
        (2) patient number to test (eg. 205)

Output: (1) saves txt file of dice score for entire brain 

Usage: test.py [patient_number]
========================================================= 

TODO
- switch to make prediction for an entire brain
- add function to calculate dice score
- save dice score output to text file
'''

# parse the command life arguments and config file
model_name = sys.argv[1]
patient_no = sys.argv[2]

config = configparser.ConfigParser()
data_path = config['paths']['processed']
for i in rang 
full_path = data_path+'data/test/pat'+patient_no+ '_'+slice_no+'_data.png'



model = load_model('Outputs/Models/Trained/' + model_name + '.h5')

data_img = Image.open(full_path)
data = np.asarray(data_img)
test_data = data.reshape(4, 240, 240)
input_data = np.zeros((240, 240, 4))
for i in range(4):
    input_data[:,:,i] = test_data[i,:,:]

test_patches = extract_patches_2d(input_data, (33,33))
prediction = model.predict_classes(test_patches)
np.save('Outputs/Predictions/{}_{}_{}.npy'.format(model_name, patient_no, slice_no), prediction)

pred_sq = prediction.reshape(208,208)
p = np.pad(pred_sq, (16, 16), mode='edge')

ones = np.argwhere(p == 1)
twos = np.argwhere(p == 2)
threes = np.argwhere(p == 3)
fours = np.argwhere(p == 4)


new = p.copy() 
for i in range(len(ones)):
    new[ones[i][0]][ones[i][1]] = 63 
for i in range(len(twos)):
    new[twos[i][0]][twos[i][1]] = 127
for i in range(len(threes)):
    new[threes[i][0]][threes[i][1]] = 191 
for i in range(len(fours)):
    new[fours[i][0]][fours[i][1]] = 255

new_img = new.astype(np.uint8)
imageio.imwrite('Outputs/Segmentations/{}_{}_{}.png'.format(model_name, patient_no, slice_no), new_img)

label_path = data_path + 'labels/test/pat' + patient_no + '_' + slice_no + '_label.png'
label_img = Image.open(label_path)
label = np.asarray(label_img)

plt.figure(figsize=(15,10))


data_figure = test_data[1]
plt.subplot(131)
plt.title('Input')
plt.imshow(data_figure, cmap='gray')

plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(label,cmap='gray')

plt.subplot(133)
plt.title('Prediction')
plt.imshow(p,cmap='gray')

plt.show()

plt.savefig('Outputs/Segmentations/{}_{}_{}_prediction.png'.format(model_name, patient_no, slice_no), bbox_inches='tight')

y = label[15:223, 15:223]
truth = y.reshape(43264,)
print(classification_report(truth, prediction, labels=[0,1,2,3,4]))





