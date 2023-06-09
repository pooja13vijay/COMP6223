#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[3]:


def convolve(image, kernel):
    # find height and length of kernel
    kernel_length = kernel.shape[0]
    kernel_height = kernel.shape[1]
    
    # find height and length of image
    img_height = image.shape[1]
    img_length = image.shape[0]
    
    # set padding length and height
    padding_length = kernel_length // 2
    padding_height = kernel_height // 2
    
    # get the number of channels
    num_channels = image.shape[2]
    
    # create empty padded 3-d image array filled with zeros 
    pad_img = np.zeros((img_length +(kernel_length - 1), img_height + (kernel_height - 1), num_channels))
    # place the image inside zero padded 3-d array
    pad_img[padding_length:-padding_length, padding_height:-padding_height, :] = image
    
    #create empty array of zeros in same shape as image
    output_img = np.zeros(image.shape)
    
    # rotate the kernel by 180 degrees
    kernel = kernel[::-1,::-1]
    
    # iterate through each channel of the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for channel in range(num_channels):
                # get current section of padded image in relation to kernel size
                sect = pad_img[i:i+kernel_length, j:j+kernel_height, channel]
                # perform convolution by multiplying kernel values and image pixel values, get the sum and normalize the sum value
                output_img[i,j, channel] = np.sum(np.multiply(sect, kernel) / 255.0)
    # return output image
    return output_img


# In[ ]:




