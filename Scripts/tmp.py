with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']

if len(sys.argv) == 1:
    print('[model name] [patient_no] [slice_no]')

model_name = sys.argv[1]
patient_no = sys.argv[2]
slice_no = sys.argv[3]

slice_img = io.imread(root+'/test/pat_'+str(patient_no)+'_'+str(slice_no)+'_strip.png')
slice_img = skimage.img_as_float(slice_img)
slice = np.array(slice_img)
slice = slice.reshape(4,240,240)

input_data = np.zeros((240,240,4))

for i in range(4):
    input_data[:,:,i] = slice[i,:,:]

input_patches = extract_patches_2d(input_data, (33,33))
model = load_model('Outputs/Models/Trained/' + model_name + '.h5')
pred = model.predict_classes(input_patches)
np.save('Outputs/Predictions/{}_{}_{}.npy'.format(model_name, patient_no, slice_no), pred)
p = pred.reshape(208, 208)
prediction = np.pad(p, (16,16), mode='edge')

label_img = io.imread(root+'/test/pat_'+str(patient_no)+'_'+str(slice_no)+'_label.png')
label_img = skimage.img_as_float(label_img)
label = np.array(label_img)

scan = slice[1]

plt.figure(figsize=(15,10))

plt.subplot(131)
plt.title('Input')
plt.imshow(scan, cmap='gray')

plt.subplot(132)
plt.title('Ground Truth')
plt.imshow(label,cmap='gray')

plt.subplot(133)
plt.title('Prediction')
plt.imshow(prediction,cmap='gray')

plt.show()

plt.savefig('Outputs/Segmentations/{}_{}_{}_prediction.png'.format(model_name, patient_no, slice_no), bbox_inches='tight')

truth = label[15:223,15:223]
































'''
=============== Processing testing =============== 

with open('config.json') as config_file:
    config = json.load(config_file)

root = config['processed']

data = io.imread(root+'/train/pat0_100_data.png').astype('float')
label = io.imread(root + '/train/pat0_100_label.png').astype('float')
#img = plt.imread(root+'/train/pat0_108.png')

x = np.array(data)
x /= 255
x = x.reshape(4,240,240)
y = np.array(label)
print(np.min(x))
print(np.max(x))





test = np.load(root + '/train/pat0_108.npy') # (240,240,5)
data = test[:,:,:4] # (240,240,4)
label = test[:,:,4] # (240,240)

strip = np.zeros((4, 240, 240))
for i in range(4):
    strip[i,:,:] = data[:,:,i]

strip = strip.reshape(960,240)
imageio.imwrite('Outputs/tmp/test_data_pat0_108.png',strip,'F')
imageio.imwrite('Outputs/tmp/test_label_pat0_108.png',label,'F')


test_strip = io.imread('Outputs/tmp/test_data_pat0_108.png').astype('float')
test_label = io.imread('Outputs/tmp/test_label_pat0_108.png').astype('float')

x = np.array(test_strip)
y = np.array(test_label)

x = x.reshape(4, 240, 240)
print('data max: ' + str(np.min(x)))
print('data min: ' + str(np.max(x)))

print('label min: ' + str(np.min(y)))
print('label max: ' + str(np.max(y)))

'''

