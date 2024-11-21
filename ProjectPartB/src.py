# CSI41033 Project Part B
# Name: Zijun Ye 
# Student Number: 300168065

# Import the required modules 
import cv2 
from deepface import DeepFace
import numpy as np
recent_predictions = []

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

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # 1. Face Detection 
    # result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    emotion = result[0]['dominant_emotion']

    recent_predictions.append(emotion)
    if len(recent_predictions) > 5:  # Keep the last 5 predictions
        recent_predictions.pop(0)
    smoothed_emotion = max(set(recent_predictions),key= recent_predictions.count)

    # Draw rectangle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Font type
    # Write these things
    # cv2.putText(frame, result[0]['dominant_emotion'], (0, 50), font, 2, (0, 255, 0), 3, cv2.LINE_4)
    cv2.putText(frame, smoothed_emotion, (0, 50), font, 2, (0, 255, 0), 3, cv2.LINE_4)
    


    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Face Detection and Analysis', frame)

    # Press any key to exit the loop
    if cv2.waitKey(1) != -1:  # Any key pressed will break the loop
        print("Key pressed. Exiting loop...")
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()


# Face detect: Neutral, Fear, Suprise, Happy, Sad, disgusting QQ