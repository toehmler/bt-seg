import numpy as np
import json
import imageio
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import imageio
#from Utils import patches
from memory_profiler import profile
import gc, sys
import random

'''
with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']
'''
def my_func():
    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['processed']
    for i in range(5):
        with np.load('{}/train/pat_{}.npz'.format(root, i)) as patient:
            data = patient['data']
            scans = data[:,:,:,:4]
            labels = data[:,:,:,4]
            print(np.min(labels))
            print(np.max(labels))
        '''
    path = '{}/train/pat_{}.dat'.format(root, 0)
    tmp = np.zeros((240,240,5), dtype='float32')
    scans = np.memmap(path, dtype='float32', mode='c', shape=(155,240,240,5))
    rand_idx = random.randint(0,154)
    tmp[:,:,:] = scans[rand_idx,:,:,:]
    print(np.min(tmp))
    print(np.max(tmp))
    '''


def test_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()





























'''
=============== Processing testing =============== 

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']

data = io.imread(root+'/train/pat0_100_data.png').astype('float')
label = io.imread(root + '/train/pat0_100_label.png').astype('float')
#img = plt.imread(root+'/train/pat0_108.png')

x = np.array(data)
x /= 255
x = x.reshape(4,240,240)
y = np.array(label)
print(np.min(x))
print(np.max(x))





test = np.load(root + '/train/pat0_108.npy') # (240,240,5)
data = test[:,:,:4] # (240,240,4)
label = test[:,:,4] # (240,240)

strip = np.zeros((4, 240, 240))
for i in range(4):
    strip[i,:,:] = data[:,:,i]

strip = strip.reshape(960,240)
imageio.imwrite('Outputs/tmp/test_data_pat0_108.png',strip,'F')
imageio.imwrite('Outputs/tmp/test_label_pat0_108.png',label,'F')


test_strip = io.imread('Outputs/tmp/test_data_pat0_108.png').astype('float')
test_label = io.imread('Outputs/tmp/test_label_pat0_108.png').astype('float')

x = np.array(test_strip)
y = np.array(test_label)

x = x.reshape(4, 240, 240)
print('data max: ' + str(np.min(x)))
print('data min: ' + str(np.max(x)))

print('label min: ' + str(np.min(y)))
print('label max: ' + str(np.max(y)))

'''

