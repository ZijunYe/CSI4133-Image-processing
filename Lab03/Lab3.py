# Lab3
# Name: Zijun Ye 
# Student Number: 300168065
import cv2  

# 1 Load input video 
video = cv2.VideoCapture('video/park.avi') # open the video file 

# Get all the original Video's data 
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
# get frame sizes 
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of frames
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of frames

# Define and create the result video by specifiy the traits 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #define the video compressed format 
result = cv2.VideoWriter('result.mp4',fourcc,fps,(frame_width,frame_height), isColor=False)


currentFrame = 0 # start with frame 0 
while currentFrame < total_frames:  # 6. Repeat the step2-step5 until the last frame of the input video.
    # 2 Obtain two consecutive frames 
    ret1, frame1 = video.read()  # First frame read
    ret2, frame2 = video.read()  # Second frame read

    # 3. Calculate the pixel intensity difference between the two consecutive
    # frames
    # Convert frames to grayscale if you need to compare pixel intensity
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the difference
    frame_diff = cv2.absdiff(gray_frame1, gray_frame2)


    # 4. Perform thresholding on the difference image to get areas of movement in binary format 
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # 5. Save the resulting frame in a new video (eg: “result.avi”)
    result.write(threshold)

    currentFrame +=2


# Open the video 
video.release() # original video
result.release() # result video
cv2.destroyAllWindows()