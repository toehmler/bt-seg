# Automatic Brain Tumor Segmentation

## Table Of Contents:
1. High Grade Gliomas
2. MRI Scans
3. Why do we need Automatic Segmentation?
4. Dataset
5. Convolutional Neural Networks
6. Our Model

## High Grade Glioma Brain Tumors

Glioma tumors are the most common type of brain tumor, comprising approximately 33% of all brain tumors. These tumors originate in 
different types of glial cells, which surround and provide support for neurons in the brain. Gliomas are classified from Grade I to 
Grade IV by their various rates of growth. While Grade I gliomas are usually removable surgically with a promising survival rate, 
high grade gliomas (glioblastomas) are one of the most deadly cancers, with only a 5% survival rate after 5 years. The current standard of 
treatment for these tumors generally consists of some combination of surgery, radiation and chemotherapy. The early detection and diagnosis 
of these tumors is crucial to long-term survival rate. This is generally done through brain imaging, including MRI, CT and PET scans. 
Of these, the most commonly used imaging technique is the MRI scan, due to its ability to non-invasively provide accurate characterizations 
of different tissue types.

## MRI Scans

Magnetic resonance imaging (MRI) scans work by applying a strong magnetic field to align protons in the brain, before using radiofrequency 
pulses to disturb the alignment. When the radiofrequency field is turned off, MRI's can measure the energy emission as protons return to 
alignment with the magnetic field. MRI scans are particularly effective at imaging soft tissue, and organs like the brain and the heart. 
MRI's visual the brain through taking a series of two-dimensional 'slices' (at 1mm increments) in one of three planes: 
coronal, sagittal and axial. In these slices, each pixel represents a 1mm^3 voxel. For the purposes of our model, 
we used slices in the axial plane because it is easier to visualize/IN BRATS DAtASET and the resolution is the HIGHEST??

## Why do we need Automatic Segmentation?

Given the huge number of slices generated by these MRI scans (we have 620 for each patient), it is incredibly laborious for a radiologist to go
label the voxels (240x240) in each slice (155 slices). This is time radiologists can use to focus on other tasks. Thus, an effective automatic segmentation method 
could provide a much more efficient alternative, saving the radiologist and the patient valuable time. Indeed, one state-of-the-art algorithm
published in 2017 can provide a segmentation between 25 seconds and 3 minutes (Havaei et al.), which is manually inconceivable.
Further, manually applying these labels require high level of expertise and are prone to human error. The use of a highly-trained convolutional 
neural network might be able to pick up small contrasts and edges that are hard to detect with the human eye.

### PICTURE OF DIFFERENT SLICES?

## Dataset

All MRI brain scans were provided by the BRATS 2015 challenge database (https://www.smir.ch/BRATS/Start2015). 
This dataset consists of 246 high-grade glioma cases and 54 low-grade glioma cases. Each scan consists of 155 slices in four different 
modalities: T1, T1 with contrast, T2 and FLAIR (each of these uses a different pulse sequence to create different pixel contrasts in 
a MR brain image). Thus, there are 620 MR images for each patient, and 186,000 images overall. Further, each patient has a fifth image 
providing the 'ground truth' labels for each pixel. In this dataset, the labels are as follows: '0' is 'non-tumor;' '1' is 'necrosis'; 
'2' is 'edema'; '3' is 'non-enhancing tumor'; '4' is 'enhancing tumor.' There is a label for each pixel in each 240x240 voxel slice, generating
8,928,000 labels for each patient, and 2,678,400,000 labels in the dataset overall (300 patients). These ground truth segmentation labels are 
manually provided by radiologists. 


## Convolutional Neural Networks

Convolutional Neural Networks (CNN) are deep learning algorithms that are commonly used for image processing, object detection and classification 
tasks. Neural networks can 'learn' through the fine tuning of large numbers of weight and biases in the network to adapt to a specific task. 
These networks are modelled after the structure of the human brain, as they consist of a series of complex layers of connections between 
artificial 'neurons,' or perceptrons. The first CNN, entitled 'AlexNet,' was created in 2012 by Geoffrey Hinton and his colleagues at the 
University of Toronto. Since then, they have been used extensively, but the application of CNN's to medical images is a very new development. 
CNN's are a well-suited tool for our task of the automatic segmentation of tumors. 



## References