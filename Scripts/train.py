from skimage import io
from glob import glob
import numpy as np

root = '/Users/treyoehmler/dev/tumors/seg/'

strip = io.imread(root + 'test.png').reshape(4, 240, 240)
label = io.imread(root + 'label_test.png')

label = label / 255 
label = label * 4

print(np.min(label))
print(np.max(label))


print(strip.shape)
print(label.shape)
