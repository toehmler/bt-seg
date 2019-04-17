from Utils import config
from Utils import patches

out_path = '/home/trey/bt-seg/Outputs/Patches/test1/'
test_patches = patches.generate_train(10, config.train_root, 33)

patches = test_patches[0]
for i, patch in enumerate(patches):
    # (4, 33, 33) -> (33, 33, 4)
    strip = np.zeros((4, 33, 33))
    for j in range(4):
        strip[j:,:] = patch[:,:,j]

    strip_img = strip.reshape(132, 33)
    strip_img = strip.astype(np.uint8)
    imageio.imwrite(out_path + '{}_patch.png'.format(i), strip_img)

# reshape patches to strips

# save as pngs
