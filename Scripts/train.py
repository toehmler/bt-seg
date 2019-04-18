from glob import glob
import numpy as np
from Utils import config
from Utils import patches
from Models import m1
from keras.utils import np_utils
import random
import sys

if len(sys.argv) == 1:
    print('training_path num_per bs epochs save_name')

training_path = sys.argv[1]
num_per = int(sys.argv[2])
bs = int(sys.argv[3])
training_epochs = int(sys.argv[4])
save_name = sys.argv[5]
'''
Command line arguments:
    - path to training data
    - number of each class of patches
    - batch size
    - epochs
    - name to save under
'''

print("Generating patches...")

training_patches = patches.generate_train(num_per, training_path, 33)

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
model.fit(x_train, y_train, batch_size=bs, epochs=training_epochs, validation_split=0.1, verbose=1) 

model.save('Outputs/Models/Trained/' + save_name + '.h5')





