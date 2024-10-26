# Introducation to Opencv

## Part A: Display an Image 
- Read Image 
	- ```cv2.imread(arg1,arg2)```
	- arg1: string representing the path of the image 
	-arg2: flag that specific how the image should be read 
	- Other method
		- ```cv2.Imread_color```
			- loads color image 
			- any transparency of images is neglected 
			- default flag 

		- ```cv2.imread_grayscale 
			- load image in grayscale mode 
		- ```cv2.imread_unchange``` 

- Display an image 
	- ```cv2.imshow(arg1,arg2)
		- arg1: window name 
		- arg2: image 

## Part B: Save an image 
	- ```cv2.imwrite(arg1,arg2)```
		arg1: string representation of file name, include image format (png, jpg)
		arg2: Image to be saved 

	- creating an image from scratch 
		- import numpy as np
		- np.zeros((height,width,3),np.unit8) 
		- use .shape to find height and width 

## Part C: Down-sample an Image 
	- Selecting one single value to represent several values 
	- Makes data more manageable 
	- reduces the dimensionality of the data, enabling faster processing of data 
	- reduces storage size of the data 

## Part D: quantize an Image 
	- Mapping of a large range of possible sample value into a smaller range of values or codes 





# Calculating the difference between two consecutive Images 

### Motion Detection 
- Frame: a sequence of images 
- Motion detection system: 
- frames that taken at different times, the difference between the frame 
- the difference between frames in intensity level of corresponding pixels 

    1. Fixed Camera on moving object 
    2. Moving Camera on Fixed object 
    3. Moving Camera on Moving object 


### Procedure 
1. Load two successive frames from same video
2. Convert RGB images to gay scale images 
3. Calculate the pixel intensity difference between two images(absolute value)
4. Perform thresholding on the difference image to get areas movement in binary format 
5. change threshold values to see different results
6. Save the best resulting images


# Video
- Initializing capture from a camera
``` cv2.VideoCapture(0);// open the default camera``` 

- Initializing capture from a video file 
``` cv2.videoCapture('video.avi"); //open the video file


# Color Based object detection 
1. Hue color-correspondence experiment (lightness)
2. Color-based object detection 

## Hue Color-correspondence experiment
- Hue is part of HSV 
- HSV: Hue Saturation, Value 
- Number range 
    - Hue: 0 to 179
    - Saturation: 0 to 255
    - Value: 0 to 255 
- https://www.geeksforgeeks.org/color-spaces-in-opencv-python/ 
*Process* 
1. Load image (folder “images”)
2. Convert image from RGB space into HSV space
```cv2.cvtColor(image, cv2.COLOR_BGR2HSV)```


3. Isolate pixels with a specific hue value 
• Use a track bar to set the Hue value H_v
• Use loops to get all the H,S,V values
    - If (H_current != H_v)
    - Then set H_current, S_current, V_current = 0

- Creation of track bar: 
```
# Create a window
cv2.namedWindow('Isolate Hue')

# Create a trackbar to adjust the Hue value
cv2.createTrackbar('Hue', 'Isolate Hue', 0, 179, nothing)
```

```
shape[0] gives the height of the image (the number of rows or pixels vertically).
shape[1] gives the width of the image (the number of columns or pixels horizontally).
shape[2] gives the number of channels (for a color image, this is usually 3: one each for Red, Green, and Blue).
```


4. Convert the image containing the isolated pixels from HSV 
space to RGB space
5. Visualize the results

**Analysis**
Experiment to see which hue-values correspond to which visible-spectrum colours in OpenCV.


*Procedure* 
1. Get the appropriate Hue value/ranges for the yellow-green square/violet square/red square from Hue colour-correspondence experiment
2. Generate the colour masks and refine the colour masks using 
• Erode function (remove the noise)
```
eroded_mask = cv2.erode(mask, kernel, iterations=1)
```
• Dilate function (fill in gaps)
```
dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
``` 

• Pay attention to the size of the kernel elements


3. Isolate the yellow-green square, the violate square, and the red square in the grid (create a track bar to select among yellow_Green(0), Violet(1), and Red(2))
4. Show the isolated pixels (in their original colour RGB) in a window
5. Show the isolated pixels (as a binary mask of all the detected pixels) in a window




