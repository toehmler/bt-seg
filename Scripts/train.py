from skimage import io
from glob import glob
import numpy as np
from Utils import config
from Utils import patches

root = '/Users/treyoehmler/dev/tumors/seg/'
data_root = config.processed_root + 'data/'
print(data_root)

training_patches = patches.generate_train(1, config.train_root, 33)
print(training_patches[0].shape)
print(training_patches[1].shape)


# load compiled model

# load training patches

# shuffle training patches

# fit model on patches

# save model 

'''

strip = io.imread(root + 'test.png').reshape(4, 240, 240)
label = io.imread(root + 'label_test.png')

label = label / 255 
label = label * 4

print(np.min(label))
print(np.max(label))


'''
