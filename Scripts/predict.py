# @author: Trey Oehmler

from PIL import Image
from keras.models import load_model
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import configparser
import sys    
import json
import matplotlib.pyplot as plt
import imageio
from skimage import io
import skimage
from sklearn.metrics import classification_report
from tqdm import tqdm
import Utils.eval as Eval

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
'''

def load_test_slice(root, pat_no, slice_no):
    slice = io.imread('{}/test/pat_{}_{}_strip.png'
                      .format(root, pat_no, slice_no))
    slice = skimage.img_as_float(slice)
    slice = np.array(slice)
    slice = slice.reshape(4, 240, 240)
    return slice

def load_test_label(root, pat_no, slice_no):
    label = io.imread('{}/test/pat_{}_{}_label.png'
                      .format(root, pat_no, slice_no))
    label = skimage.img_as_float(label)
    label = np.array(label)
    return label

def generate_test_patches(root, pat_no, slice_no):
    slice = load_test_slice(root, pat_no, slice_no)
    input_data = np.zeros((240, 240, 4))
    for mod in range(4):
        input_data[:,:,mod] = slice[mod,:,:]
    patches = extract_patches_2d(input_data, (33,33))
    return patches

def generate_prediction_img(model_name, pat_no, slice_no, prediction):
    slice = load_test_slice(root, pat_no, slice_no)
    label = load_test_label(root, pat_no, slice_no)
    scan = slice [1]
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.title('Input')
    plt.imshow(scan, camp='gray')
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(label,cmap='gray')
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(prediction,cmap='gray')
    plt.show()
    plt.savefig('Outputs/Segmentations/{}_{}_{}_prediction.png'.
                format(model_name, pat_no, slice_no), 
                bbox_inches='tight')

def show_slice_dice(root, pat_no, slice_no, prediction):
    label = load_test_label(root, pat_no, slice_no)
    Eval.slice_dice(prediction, label)


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    root = config['processed']

    if len(sys.argv) == 1:
        print('[model name] [patient_no] [slice_no]')

    model_name = sys.argv[1]
    pat_no = sys.argv[2]
    slice_no = sys.argv[3]

    patches = generate_test_patches(root, pat_no, slice_no)
    model = load('Outputs/Models/Trained/{}.h5'.format(model_name))
    prediction = model.predict(patches, verbose=1)
    np.save('Outputs/Predictions/{}_{}_{}.npy'
            .format(model_name, pat_no, slice_no))
    prediction = prediction.reshape(208, 208)
    prediction = np.pad(prediction, (16, 16), mode='edge')
    generate_prediction_img(model_name, pat_no, slice_no, prediction) 
    show_slice_dice(root, pat_no, slice_no, prediction)

'''
slice_img = io.imread(root+'/test/pat_'+str(patient_no)+'_'+str(slice_no)+'_strip.png')
slice_img = skimage.img_as_float(slice_img)
slice = np.array(slice_img)
slice = slice.reshape(4,240,240)

input_data = np.zeros((240,240,4))

for i in range(4):
    input_data[:,:,i] = slice[i,:,:]

input_patches = extract_patches_2d(input_data, (33,33))
model = load_model('Outputs/Models/Trained/' + model_name + '.h5')
pred = model.predict(input_patches, verbose=1)
#np.save('Outputs/Predictions/{}_{}_{}.npy'.format(model_name, patient_no, slice_no), pred)
p = pred.reshape(208, 208)
prediction = np.pad(p, (16,16), mode='edge')

label_img = io.imread(root+'/test/pat_'+str(patient_no)+'_'+str(slice_no)+'_label.png')
label_img = skimage.img_as_float(label_img)
label = np.array(label_img)

scan = slice[1]

plt.figure(figsize=(15,10))

plt.subplot(131)
plt.title('Input')
plt.imshow(scan, cmap='gray')

plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(label,cmap='gray')

plt.subplot(133)
plt.title('Prediction')
plt.imshow(prediction,cmap='gray')

plt.show()

plt.savefig('Outputs/Segmentations/{}_{}_{}_prediction.png'.format(model_name, patient_no, slice_no), bbox_inches='tight')

truth = label[15:223,15:223]
'''





