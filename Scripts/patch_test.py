from Utils import config
from Utils import patches
import numpy as np
import imageio


'''
Description of tests:
    - test1: no restriction placed on len(np.argwhere(patch == 0))
    - test2: resample if len(np.argwhere(patch == 0)) > (33*33)
    - test3: resample if len(np.argwhere(patch == 0)) > ((33*33) / 4)

'''


patches_per = 100

out_path = '/home/trey/bt-seg/Outputs/Patches/test3/'
test_patches = patches.generate_train(patches_per, config.train_root, 33)

patches = test_patches[0]

for i in range(len(test_patches[0])):
    patch = test_patches[0][i]
    label = test_patches[1][i]
    strip = np.zeros((4, 33, 33))
    for j in range(4):
        strip[j,:,:] = patch[:,:,j]
    
    strip_img = strip.reshape(132,33)
    s = strip_img.astype(np.uint8)
    imageio.imwrite(out_path + 'label_{}_patch_{}.png'.format(label, i), s)
