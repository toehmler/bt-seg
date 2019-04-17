import numpy as np
import imageio
from PIL import Image
from sklearn.metrics import classification_report, precision_score, recall_score


# -*- coding: utf-8 -*-
"""test.py 
"""

root = '/Users/treyoehmler/dev/tumors/seg/'

prediction = np.load(root + 'Outputs/210_55_2.npy')
pred = prediction.reshape(208, 208)
p = np.pad(pred, (16, 16), mode='edge')

label_img = Image.open(root  + 'pat200_50_label.png')
label = np.asarray(label_img)

ones = np.argwhere(p == 1)
twos = np.argwhere(p == 2)
threes = np.argwhere(p == 3)
fours = np.argwhere(p == 4)

new = p.copy() 
for i in range(len(ones)):
    new[ones[i][0]][ones[i][1]] = 63
for i in range(len(twos)):
    new[twos[i][0]][twos[i][1]] = 127
for i in range(len(threes)):
    new[threes[i][0]][threes[i][1]] = 191
for i in range(len(fours)):
    new[fours[i][0]][fours[i][1]] = 255

new_img = new.astype(np.uint8)
imageio.imwrite('200_50.png', new_img)

y = label[15:223, 15:223]
truth = y.reshape(43264,)
print(classification_report(truth, prediction, labels=[0,1,2,3,4]))







