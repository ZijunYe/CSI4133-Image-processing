# Lab2
# Name: Zijun Ye 
# Student Number: 300168065
import cv2  

#1 load image 
carFrame1 = cv2.imread('images01/car1.bmp')
carFrame2 = cv2.imread('images01/car2.bmp')

cv2.imshow('Frame1',carFrame1)
cv2.imshow('Frame2',carFrame2)

#2 Convert RGB image to gray scale image 
# use method cvtColor(src,bwsrc, cv::COLOR_RGB2GRAY)
grayCarFrame1 = cv2.cvtColor(carFrame1,cv2.COLOR_RGB2GRAY)
grayCarFrame2 = cv2.cvtColor(carFrame2,cv2.COLOR_RGB2GRAY)

cv2.imshow('grayFrame1',grayCarFrame1)
cv2.imshow('grayFrame2',grayCarFrame2)

#3 Calculate the pixel intensity difference between two images (absolute value)
# can either be subtract OR absdiff(give absolute value)
frame_difference = cv2.absdiff(grayCarFrame1,grayCarFrame2)

#create a window to display 
cv2.namedWindow('Thresholded Image')

#4 Perform thresholding on the difference image to get areas of movement in binary form
# threshold(src input image, threshold value, new value value,max value, thresholding type )
def update_threshold(val):
    thresh_value, thresholded = cv2.threshold(frame_difference, val , 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresholded Image', thresholded)

#5 create the track bar ("track bar name", "window name", initial value, max value, call back function)
cv2.createTrackbar('Threshold', 'Thresholded Image', 30, 255, update_threshold)

update_threshold(30)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()