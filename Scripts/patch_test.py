from Utils import patches
import numpy as np
import imageio


root = '/storage/s12qr5ep/data/s'
out_path = '/storage/s12qr5ep/patches/1'


#root = '/Users/treyoehmler/dev/tumors/data/tmp'
#out_path = '/Users/treyoehmler/dev/tumors/patches/1'

patches.save_training(root, 10, 33, out_path)

'''
x, y = patches.load_training(out_path, 33)
print(y[1])
print(x[1])
'''






