import os
import random
import numpy as np
from skimage import io
from PIL import Image
import imageio
from tqdm import tqdm
from glob import glob
import skimage
from keras.utils import np_utils


# -*- coding: utf-8 -*-
"""patches.py 
"""

def find_bounds(center, size):
    '''
    finds the bounding indices for a patch
    input  (1) tuple 'center': indices of center pix to find bounds for
    output (1) tuple 'bounds': indices of patch to be generated
    '''
    top = center[0] - ((size - 1) / 2)
    bottom = center[0] + ((size + 1) / 2)
    left = center[1] - ((size - 1) / 2)
    right = center[1] + ((size + 1) / 2)
    bounds = np.array([top, bottom, left, right], dtype = int)
    return bounds

def generate_train(num, root, size):
    """
    Generates a set of patches (num of each class)
    Output: num * 5 patches
    """
    patients = glob(root + '/train/*.npy')

    patches = []
    labels = []

    for i in tqdm(range(num)):
        class_label = 0
        while class_label < 5:
            # pick random patient
            path = random.choice(patients)
            print(path)
            data = np.load(path)
            # pick random slice
            slice = random.choice(data)
            y = slice[:,:,4]
            x = slice[:,:,:4]
            # resample if label not in slice
            if len(np.argwhere(y == class_label)) < 10:
                continue;
            center = random.choice(np.argwhere(y == class_label))
            bounds = find_bounds(center, size)
            patch = x[bounds[0]:bounds[1],bounds[2]:bounds[3],:]
            # resample if patch is on an edge
            if patch.shape != (size, size, 4):
                continue
            # resample if patch is > 75% background
            if len(np.argwhere(patch == 0)) > (size * size):
                continue
            patches.append(patch)
            labels.append(class_label)
            class_label += 1
    labels = np.array(labels).astype(np.float16)
    patches = np.array(patches)
    return patches, labels

'''
    for i in tqdm(range(num)):
        class_label = 0
        while class_label < 5:
            # pick a random label
            y_path = random.choice(y_paths)
            y_img = io.imread(y_path).astype('float') 
            y = np.array(y_img)
            # resample if label is not in random choice
            if len(np.argwhere(y == class_label)) < 10:
                continue
            # load corresponding strip and reshape
            slice_path = y_path[:-9] + 'data.png'
            slice_img = io.imread(slice_path)
            slice_img = skimage.img_as_float(slice_img)
            slice = np.array(slice_img)
            slice = slice.reshape(4,240,240)
            # find patch boundaries
            center = random.choice(np.argwhere(y == class_label))
            bounds = find_bounds(center, size)
            patch = slice[:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            # resample if patch is near an edge
            if patch.shape != (4, size, size):
               continue
            # resample if patch is > 75% background
            if len(np.argwhere(patch == 0)) > (size * size):
                continue
            # reshape patch (4,240,240) ==> (240,240,4) 
            x = np.zeros((size,size,4))
            for z in range(4):
                x[:,:,z] = patch[z,:,:]
            patches.append(x)
            labels.append(float(class_label))
            class_label += 1 
    labels = np.array(labels)
    y = np_utils.to_categorical(labels)
    return np.array(patches), y
'''

'''

    for i in range(5):
        print("Finding patches: " + str(i))
        count = 0
        while count < num:
            slice_path = random.choice(label_paths)
            label_img = Image.open(root + '/labels/train/' + slice_path)
            slice_label = np.asarray(label_img)
            if len(np.argwhere(slice_label == i)) < 10:
                continue

            center = random.choice(np.argwhere(slice_label == i))
            bounds = find_bounds(center, size)
            data_path = root + '/data/train/' + slice_path[:-9] + 'data.png'
            slice_img = Image.open(data_path)
            slice_data = np.asarray(slice_img)
            slice_data = slice_data.reshape(4, 240, 240)
            p = slice_data[:,bounds[0]:bounds[1], bounds[2]:bounds[3]]

            if p.shape != (4, size, size):
                continue

            if len(np.argwhere(p == 0)) > (33 * 33): 
                continue

            patch = np.zeros((size,size,4))
            for j in range(4):
                patch[:,:,j] = p[j,:,:]

            patches.append(patch)
            labels.append(i)
            count += 1

    return np.array(patches), np.array(labels)
        
'''
    

#labels_paths = os.listdir(root + 'data/train/')

# find balanced classes ( call find_patch())


# randomly select a path

# check if class label is in slice

# check that patch is of the right size

# check that patch is not 25% empty 


    





    





