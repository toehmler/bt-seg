import numpy as np
import json
import imageio
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from Utils import patches

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']

x, y = patches.generate_train(1, root, 33)
print(x)
print(y)















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

