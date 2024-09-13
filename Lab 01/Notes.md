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