import os
import random
import numpy as np
from skimage import io
from PIL import Image
import imageio


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
    label_paths = os.listdir(root + 'labels/train/')

    patches = []
    labels = []

    for i in range(5):

        count = 0
        while count < num:
            slice_path = random.choice(label_paths)
            label_img = Image.open(root + 'labels/train/' + slice_path)
            slice_label = np.asarray(label_img)
            if len(np.argwhere(slice_label == i)) < 10:
                continue

            center = random.choice(np.argwhere(slice_label == i))
            bounds = find_bounds(center, size)
            data_path = root + 'data/train/' + slice_path[:-9] + 'data.png'
            slice_img = Image.open(data_path)
            slice_data = np.asarray(slice_img)
            slice_data = slice_data.reshape(4, 240, 240)
            p = slice_data[:,bounds[0]:bounds[1], bounds[2]:bounds[3]]
            if p.shape != (4, size, size):
                continue

            patch = np.zeros((size,size,4))
            for j in range(4):
                patch[:,:,j] = p[j,:,:]

            if len(np.argwhere(patch == 0)) > (3 * ((size * size) / 4)):

                continue
            patches.append(patch)
            labels.append(i)
            count += 1

    return np.array(patches), np.array(labels)
        

    

    #labels_paths = os.listdir(root + 'data/train/')

    # find balanced classes ( call find_patch())


    # randomly select a path

    # check if class label is in slice

    # check that patch is of the right size

    # check that patch is not 25% empty 


    





    





