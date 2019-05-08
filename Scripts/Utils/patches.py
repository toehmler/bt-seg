import os, gc, ctypes
import random
import numpy as np
from skimage import io
from PIL import Image
import imageio
from tqdm import tqdm
from glob import glob
import skimage
from keras.utils import np_utils
import json

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


def generate_training(root, num, size):
    label_paths = glob(root + '/train/*_label.png')
    patches = []
    labels = []
    for i in tqdm(range(num)):
        class_label = 0
        while class_label < 5:
            path = random.choice(label_paths)
            label = imageio.imread(path)
            if len(np.argwhere(label == class_label)) < 10:
                continue
            strip_path = path.replace('_label.png', '_strip.png')
            strip = imageio.imread(strip_path)
            strip = skimage.img_as_float(strip)
            strip = strip.reshape(4, 240, 240)
            center = random.choice(np.argwhere(label == class_label))
            bounds = find_bounds(center, size)
            sample = strip[:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            if sample.shape != (4, size, size):
                continue
            if len(np.argwhere(sample == 0)) > (size * size):
                continue
            patch = np.zeros((size, size, 4))
            for mod in range(4):
                patch[:,:,mod] = sample[mod,:,:]
            patches.append(patch)
            labels.append(class_label)
            class_label += 1
    labels = np.array(labels)
    y = np_utils.to_categorical(labels)
    return np.array(patches), y

def load_training(root, num):
    paths = glob(root + '/*.png')
    patches = []
    labels = []
    for i in tqdm(range(num)):
        path = paths[i]
        label = float(path[-5])
        patch_img = imageio.imread(path)
        patch_img = skimage.img_as_float(patch_img)
        size = patch_img.shape[1]
        patch_img = patch_img.reshape(4, size, size)
        patch = np.zeros((size, size, 4)) 
        for mod in range(4):
            patch[:,:,mod] = patch_img[mod,:,:]
        patches.append(patch)
        labels.append(label)
    labels = np.array(labels)
    y = np_utils.to_categorical(labels)
    return np.array(patches), y
       
if __name__ == '__main__':
    root = '/Users/treyoehmler/dev/tumors/patches/1/1';
    x, y = load_training(root, 10)
    for label in y:
        print(label)
    for patch in x:
        print(patch)

    new = '/Users/treyoehmler/dev/tumors/data/tmp';
    a, b = generate_training(new, 2, 33)
    for label in b:
        print(label)
    for patch in a:
        print(patch)


    



'''

def generate_train_batch(start, num_patients, num_per, root, size):





    batch_patches = np.zeros((num_patients,5*num_per,size,size,4),dtype='float32')
    batch_labels = []
    for i in tqdm(range(num_patients)):
        path = '{}/train/pat_{}.dat'.format(root, start + i)
        pat_patches, pat_labels = generate_patient_patches(path, num_per, size)
        batch_patches[i] = pat_patches
        batch_labels.append(pat_labels)
    return batch_patches.reshape(num_patients*5*num_per,size,size,4), np.array(batch_labels)         

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    root = config['processed']
    x, y = generate_train_batch(3, 5, 75, root, 33)

    patches = np.zeros((num_per*5, size, size 4), dtype='float32')
    labels = np.zeros((num_per, size, size), dtype='float32')
    scans = np.memmap(path, dtype='float32', mode='r', shape=(155,240,240,5))
    for i in tqdm(range(num_per)):
        class_label = 0
        while class_label < 5:
            idx = random.randint(0,154)
            slice_label = scans[idx,:,:,4]
            if len(np.argwhere(slice_label == class_label)) < 10:
                continue
            center = random.choice(np.argwhere(slice_label == class_label))
            bounds = find_bounds(center)
            patch = scans[idx,bounds[0]:bounds[1],bounds[2]:bounds[3],:4]
            if patch.shape != (size,size,4):
                continue
            if len(np.argwhere(patch == 0)) > (size * size):
                continue
            for j in range(4):
                if np.max(patch[:,:,j]) != 0:
                    patch[:,:,j] /= np.max(patch[:,:,j])
            
            patches.append(patch)
            labels.append(class_label)
            class_label += 1
    del scans
    labels = np.array(labels).astype(np.float16)
    labels = np_utils.
======================================== 
TODO:
    - finish batch patch generation
    - train model 1 again using new format
    - predict a couple slices (predict.py)
    - predict a few brains (test.py)
    - get some dice scokre



def generate_train_batch(start, num_patients, num_per, root, size):
    for i in tqdm(range(num_patients)):
        path = '{}/train/pat_{}.dat'.format(root, start + i)

def generate_train(num, num_per_class, root, size):
    Generates a set of patches (num of each class)
    Output: num * 5 patches
    patients = glob(root + '/train/*.dat')

    patches = []
    labels = []

    for i in tqdm(range(num)):
        scans = np.memmap(patients[i], dtype='float32', mode='c', shape=(155,240,240,5))
        for z in range(num_per_class):
            class_label = 0
            while class_label < 5:
                # pick random slice
                idx = random.randint(0,154)
                if len(np.argwhere(scans[idx,:,:,4] == class_label)) < 10:




                slice = random.choice(scans)
                y = slice[:,:,4]
                x = slice[:,:,:4]
                # resample if label is not in slice
                if len(np.argwhere(y == class_label)) < 10:
                    continue
                center = random.choice(np.argwhere(y == class_label))
                bounds = find_bounds(center, size)
                patch = x[bounds[0]:bounds[1],bounds[2]:bounds[3],:]
                # resample if patch is on an edge
                if patch.shape != (size, size, 4):
                    continue
                # resample if patch is > 75% background
                if len(np.argwhere(patch == 0)) > (size * size):
                    continue
                # set pixel intensity between 0 and 1
                for j in range(4):
                    if np.max(patch[:,:,j]) != 0:
                        patch[:,:,j] /= np.max(patch[:,:,j])
                patches.append(patch)
                labels.append(class_label)
                class_label += 1
        del scans
    labels = np.array(labels).astype(np.float16)
    labels = np_utils.to_categorical(labels)
    patches = np.array(patches)
    return patches, labels




    for i in range(num):
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

'''for

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


    





    





