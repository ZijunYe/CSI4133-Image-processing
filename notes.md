## Lab2 : Calculating the difference between two consecutive Images 

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
6. Save the best resulting image 