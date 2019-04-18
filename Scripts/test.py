import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.image import extract_patches_2d
import imageio
from PIL import Image
from sklearn.metrics import classification_report, precision_score, recall_score
import sys
import matplotlib.pyplot as plt

''' 
Command line arguments:
    - path to data
    - name of model (eg m1_4)
    - patient no  (for prediction)
    - slice no (for prediction)
'''

# -*- coding: utf-8 -*-
"""test.py 
"""
data_path = sys.argv[1]
model_name = sys.argv[2]
patient_no = sys.argv[3]
slice_no = sys.argv[4]

#model = load_model('Outputs/Models/Trained/' + model_name + '.h5')

full_path = data_path + 'data/test/pat' + patient_no + '_' + slice_no + '_data.png'
data_img = Image.open(full_path)
data = np.asarray(data_img)
test_data = data.reshape(4, 240, 240)
input_data = np.zeros((240, 240, 4))
for i in range(4):
    input_data[:,:,i] = test_data[i,:,:]

test_patches = extract_patches_2d(input_data, (33,33))
prediction = model.predict_classes(test_patches)
np.save('/home/yb/bt-seg/Outputs/Predictions/{}_{}_{}.npy'.format(model_name, patient_no, slice_no), prediction)

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

plt.subplot(131)
plt.title('Input')
plt.imshow(data, cmap='gray')

plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(label,cmap='gray')

plt.subplot(133)
plt.title('Prediction')
plt.imshow(p,cmap='gray')

plt.show()

plt.savefig('Outputs/Segmentations/{}_{}_{}_prediction.png'.format(model_name, patient_no, slice_no))

y = label[15:223, 15:223]
truth = y.reshape(43264,)
print(classification_report(truth, prediction, labels=[0,1,2,3,4]))



'''


