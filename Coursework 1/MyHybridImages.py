#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


def makeGaussianKernel(sigma):
    
    # set the filter size
    filter_size = np.floor(8*sigma+1)
    # if filter size is even, make it odd by adding 1
    if (filter_size % 2 == 0) : 
        filter_size = filter_size +1
    # create empty filter filled with zeros
    filters = np.zeros((int(filter_size), int(filter_size)), np.float32)
    
    # find center of filter
    center = (filter_size - 1) / 2
    # iterate over filters
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            # calculate values in each position of kernel
            expVal = (i - center) ** 2 + (j - center) ** 2 
            filters[i,j] = np.exp(-expVal / (2 * sigma ** 2))
    
    # normalize filter values and return filter
    return filters / np.sum(filters)


# In[4]:


def myHybridImages(lowImage, lowSigma, highImage, highSigma):
    
    # Add extra dimension if low image is grayscale
    if (len(lowImage.shape) < 3) :
        lowImage = np.expand_dims(lowImage, axis=-1)
    
    # add extra dimension if highimage is grayscale
    if (len(highImage.shape) < 3) :
        highImage = np.expand_dims(highImage, axis=-1)
    
    # Make gaussian kernel for creating low pass filter with sigma value lowSigma
    low_kernel = makeGaussianKernel(lowSigma)
    # convolve gaussian kernel with lowImage to create low pass image
    low_pass_img = convolve(lowImage, low_kernel)
    # Make gaussian kernel for high pass filter with sigma value highSigma
    high_kernel = makeGaussianKernel(highSigma)
    # convolve gaussian kernel with highImage to create low pass image
    low_pass_img2 = convolve(highImage, high_kernel)
    # get the high pass image by subtracting the highImage from low pass image 
    high_pass_img2 = low_pass_img2 - (highImage /255.0)
    #plt.imshow(((high_pass_img2+0.5)*255.0).astype(np.uint8))
    #plt.show()
    # create hybrid images by combining the low pass image and high pass image
    hybrid = low_pass_img + high_pass_img2
    # return the hybrid image
    return hybrid


# In[ ]:




