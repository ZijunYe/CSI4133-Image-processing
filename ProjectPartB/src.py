# CSI41033 Project Part B
# Name: Zijun Ye 
# Student Number: 300168065

# import the required modules 
import cv2 
from deepface import DeepFace

# Open webcam + save the video after close the window 
# Open the default camera
cam = cv2.VideoCapture(0)


# Haar Cascade File Path 
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # 1. Face Detection 
    result= DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    #draw rectangle
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=facecascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    font=cv2.FONT_HERSHEY_SIMPLEX
    
    #font type
    #write these things
    cv2.putText(frame,result[0]['dominant_emotion'],(0,50),font,2,(0,255,0),3,cv2.LINE_4);

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Face Detection and Analysis', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()


# Face detect: Neutral, Fear, Suprise, Happy, Sad, disgusting QQ