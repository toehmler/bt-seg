from skimage import io
from glob import glob
import numpy as np
from Utils import config
from Utils import patches
from Models import m1
from keras.utils import np_utils
import random

root = '/Users/treyoehmler/dev/tumors/seg/'
data_root = config.processed_root + 'data/'
print(data_root)


training_patches = patches.generate_train(1, config.train_root, 33)

patches = training_patches[0]
labels = np_utils.to_categorical(training_patches[1])

shuffle = list(zip(patches, labels))
np.random.shuffle(shuffle)
X, Y = zip(*shuffle)

x_train = np.array(X)
y_train = np.array(Y)

model = m1.compile()
model.fit(x_train, y_train, batch_size=128, epochs=7, validation_split=0.1, verbose=1) 

#model.save('/home/trey/seg/Outputs/Models/m1.h5')





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
