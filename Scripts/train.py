from glob import glob
import numpy as np
from Utils import patches
from Models import m1
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import sys
import configparser
import json
from keras.models import load_model
from Models import two

'''
==================== train.py ==================== 
Trains a model on processed data and save .h5 file
Args:   (1) number of patches per class to train on
        (2) batch size
        (3) number of epochs to train for
        (4) name to save model under
Usage: train.py [num_per] [batch_size] [epochs] [name]
================================================== 
'''

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']

#root = '/home/yb/soup/patches'
#root = '/storage/s12qr5ep/patches/1'



if len(sys.argv) == 1:
    print('num_per bs epochs model_name')

num_per = int(sys.argv[1])
bs = int(sys.argv[2])
training_epochs = int(sys.argv[3])
save_name = sys.argv[4]
num_iters = int(sys.argv[5])

for i in range(num_iters):
#    model = load_model('Outputs/Models/Trained/{}.h5'.format(save_name))

    model = m1.compile()
#    model = two.compile((33,33,4))
    patches, labels = patches.generate_training(root, num_per, 33)
    shuffle = list(zip(patches, labels))
    np.random.shuffle(shuffle)
    x_train, y_train = zip(*shuffle)
    x = np.array(x_train)
    y = np.array(y_train)

    print(model.summary())
    #x, y = patches.load_training(root, num_per, i)
    model.fit(x, y, batch_size=bs, 
              epochs=training_epochs,
              validation_split=0.1,
              verbose=1)
    model.save('Outputs/Models/Trained/{}.h5'.format(save_name))


'''
#x, y = patches.generate_training(root, num_per, 33)
x, y = patches.load_training(root, 10000)


print('using basic model')

model = load_model
model = m1.compile()
print(model.summary())

# reformat paramters
es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(filepath="Outputs/Models/Trained/"+save_name+"{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

history = model.fit(x, y, batch_size=bs, epochs=training_epochs, validation_split=0.1, verbose=1, callbacks=[checkpointer])

model.save('Outputs/Models/Trained/' + save_name + '.h5')

with open('Outputs/Models/Trained/' + save_name + '.json', 'w') as f:
    json.dump(history.history, f)




'''

'''


for i in range(32):
    



for i in range(32):
    patch_group = patches.generate_train_batch(start=i*5,end=5
for i in range(32):

    training_patches = patches.generate_train_batch(start-i*5, 5, 75, root, 33)


training_patches = patches.generate_train(160, 75, root, 33)
shuffle = list(zip(training_patches))

np.random.shuffle(shuffle)

x_train, y_train = zip(*shuffle)
x = np.array(x_train)
y = np.array(y_train)

model = m1.compile()
print(model.summary())

es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(filepath="Outputs/Models/Trained/"+save_name+"{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
history = model.fit(x, y, batch_size=bs, epochs=training_epochs, validation_split=0.1, verbose=1, callbacks=[checkpointer])
model.save('Outputs/Models/Trained/' + save_name + '.h5')
with open('Outputs/Models/Trained/' + save_name + '.json', 'w') as f:
    json.dump(history.history, f)


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
'''





