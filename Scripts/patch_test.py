from Utils import config
from Utils import patches
import numpy as np
import imageio


patches_per = 10

out_path = '/home/trey/bt-seg/Outputs/Patches/test1/'
test_patches = patches.generate_train(patches_per, config.train_root, 33)

patches = test_patches[0]

for i in range(len(test_patches[0])):
    patch = test_patches[0][i]
    label = test_patches[1][i]
    strip = np.zeros((4, 33, 33))
    for j in range(4):
        strip[j,:,:] = patch[:,:,j]
    
    strip_img = strip.reshape(132,33)
    imageio.imwrite(out_path + 'label_{}_patch_{}.png'.format(label, i), strip_img)

'''
for i, patch in enumerate(patches):
    # (4, 33, 33) -> (33, 33, 4)
    strip = np.zeros((4, 33, 33))
    for j in range(4):
        strip[j:,:] = patch[:,:,j]
    
    strip_img = strip.reshape(132, 33)
    print(np.min(strip_img))
    print(np.max(strip_img))
    print(strip_img.shape)
    label = i // 5
#    strip_img = strip.astype(np.uint8)
    imageio.imwrite(out_path + 'label{}_patch{}.png'.format(label, i), strip_img)

# reshape patches to strips

# save as pngs
'''
