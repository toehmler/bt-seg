# Author: Trey Oehmler
import numpy as np
import json
import imageio

def unique_rows(a):
    '''
    helper function to get unique rows from 2D numpy array
    '''
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def slice_dice(prediction, gt):

    # dice coef of total image
    total = (len(np.argwhere(prediction == gt)) * 2.) / (2 * 240 * 240)
    # dice coef of entire non-background image
    gt_tumor = np.argwhere(gt != 0)
    seg_tumor = np.argwhere(prediction != 0)
    combo = np.append(seg_tumor, gt_tumor, axis = 0)
    core_edema = unique_rows(combo) 
    print(core_edema.shape)
    gt_c, seg_c = [], [] # predicted class of each
    for i in core_edema:
        gt_c.append(gt[i[0]][i[1]])
        seg_c.append(prediction[i[0]][i[1]])
    tumor_assoc = len(np.argwhere(np.array(gt_c) == np.array(seg_c))) / float(len(core_edema))
    tumor_assoc_gt = len(np.argwhere(np.array(gt_c) == np.array(seg_c))) / float(len(gt_tumor))

# dice coef advancing tumor
    adv_gt = np.argwhere(gt == 4)
    gt_a, seg_a = [], [] # classification of
    for i in adv_gt:
        gt_a.append(gt[i[0]][i[1]])
        seg_a.append(prediction[i[0]][i[1]])
    gta = np.array(gt_a)
    sega = np.array(seg_a)
    adv = float(len(np.argwhere(gta == sega))) / len(adv_gt)

# dice coef core tumor
    noadv_gt = np.argwhere(gt == 3)
    necrosis_gt = np.argwhere(gt == 1)
    live_tumor_gt = np.append(adv_gt, noadv_gt, axis = 0)
    core_gt = np.append(live_tumor_gt, necrosis_gt, axis = 0)
    gt_core, seg_core = [],[]
    for i in core_gt:
        gt_core.append(gt[i[0]][i[1]])
        seg_core.append(prediction[i[0]][i[1]])
    gtcore, segcore = np.array(gt_core), np.array(seg_core)
    core = len(np.argwhere(gtcore == segcore)) / float(len(core_gt))

    print(' ')
    print('Region_______________________| Dice Coefficient')
#    print('Total Slice__________________| {0:.2f}'.format(total))
#    print('No Background gt_____________| {0:.2f}'.format(tumor_assoc_gt))
#    print('No Background both___________| {0:.2f}'.format(tumor_assoc))
    print('Whole Tumor__________________| {0:.2f}'.format(tumor_assoc))
    print('Advancing Tumor______________| {0:.2f}'.format(adv))
    print('Core Tumor___________________| {0:.2f}'.format(core))

def binary_dice(prediction, truth):
    num = np.sum(np.multiply(prediction, truth))
    denom = prediction.sum() + truth.sum()
    if denom == 0:
        return 1
    else:
        return 2.0*num / denom

def whole_dice(prediction, truth):
    return binary_dice(prediction > 0, truth > 0)

def enhancing_dice(prediction, truth):
    return binary_dice(prediction == 4, truth == 4) 

def core_dice(prediction, truth):
    seg = np.copy(prediction)
    gt = np.copy(truth)
    seg[seg == 2] = 0
    gt[gt == 2] = 0
    return binary_dice(seg > 0, gt > 0)




if __name__ == "__main__":

    with open('config.json') as config_file:
        config = json.load(config_file)
    
    root = config['processed']
    pred = np.load('trey_4_211_60.npy')
    pred = pred.reshape(208,208)
    pred = np.pad(pred, (16, 16), mode='edge')
    label = imageio.imread('/Users/treyoehmler/dev/tumors/data/tmp/test/pat_211_60_label.png'.format(root))

    wt = whole_dice(pred, label)
    et = enhancing_dice(pred, label)
    ct = core_dice(pred, label)
    print('whole:      {0:.2f}'.format(wt))
    print('enhancing:  {0:.2f}'.format(et))
    print('core:       {0:.2f}'.format(ct))

    slice_dice(pred, label)



