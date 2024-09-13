# Lab1 
# Name: Zijun Ye 
# Student Number: 300168065
import cv2 

#load image 
imgOriginal = cv2.imread('field.jpg')

# Task1 display the image 
cv2.imshow('Task1 original Image',imgOriginal)


# find out the original Image height and width 
# 0 first shape element is height 
# 1 second shape element is width
# 2 thrid shape element is number of channels (3 colored image: RGB)
original_height = imgOriginal.shape[0] 
original_width = imgOriginal.shape[1]

# Task2 Down-sampled image 
downSample_height = original_height // 4
downSample_width = original_width // 4

# create a down-sampled image by resize 
downsampled_img = cv2.resize(imgOriginal, (downSample_width, downSample_height), interpolation=cv2.INTER_LINEAR)

#Upsample the image back to original dimensions
upsampled_img = cv2.resize(downsampled_img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

cv2.imshow('Task2 downsampled Image',downsampled_img)
cv2.imshow('Task2 unsampled Image', upsampled_img)


## Task3 Quantize an Image 
quantized_image = (imgOriginal // 32) * 32 

cv2.imshow('Task3 quantized image', quantized_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()