from PIL import Image
from keras.models import load_model
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from Utils import config

# -*- coding: utf-8 -*-
"""predict.py 
"""

data_img = Image.open(config.train_root + 'data/test/pat200_50_data.png')
test_data = np.asarray(data_img)
test_data = test_data.reshape(4, 240, 240)

data = np.zeros((240, 240, 4))
for i in range(4):
    data[:,:,i] = test_data[i,:,:]

test_patches = extract_patches_2d(data, (33,33))

model = load_model('/home/trey/bt-seg/Outputs/Models/Trained/m1.h5')

prediction = model.predict_classes(test_patches)
np.save('210_55.npy', y_pred)








