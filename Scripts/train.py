from glob import glob
import numpy as np
from Utils import config
from Utils import patches
from Models import m1
from keras.utils import np_utils
import random

root = '/Users/treyoehmler/dev/tumors/seg/'

print("Generating patches...")
training_patches = patches.generate_train(12000, config.train_root, 33)

patches = training_patches[0]
labels = np_utils.to_categorical(training_patches[1])

shuffle = list(zip(patches, labels))
np.random.shuffle(shuffle)
X, Y = zip(*shuffle)

x_train = np.array(X)
y_train = np.array(Y)


print("Training...")
model = m1.compile()
print(model.summary())
model.fit(x_train, y_train, batch_size=128, epochs=6, validation_split=0.1, verbose=1) 

model.save('/home/trey/bt-seg/Outputs/Models/Trained/m1_2.h5')





