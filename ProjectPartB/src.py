# CSI41033 Project Part B
# Name: Zijun Ye 
# Student Number: 300168065

# Import the required modules 
import cv2 
from deepface import DeepFace
import numpy as np

# Open webcam + save the video after close the window 
# Open the default camera
cam = cv2.VideoCapture(0)
# Set camera resolution to higher quality (e.g., 1920x1080)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Haar Cascade File Path 
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, 10.0, (frame_width, frame_height))

recent_predictions = []

try:
    while True:
        ret, frame = cam.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Face Detection 
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # result = DeepFace.analyze(frame, actions=['emotion','race','gender','age'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']

        recent_predictions.append(emotion)
        if len(recent_predictions) > 5:  # Keep the last 5 predictions
            recent_predictions.pop(0)
        smoothed_emotion = max(set(recent_predictions), key=recent_predictions.count)

        # Draw rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = facecascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_color = (0, 255, 0)

        # Calculate text size
        text_size = cv2.getTextSize(smoothed_emotion, font, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2  # Center horizontally
        text_y = 50  # Position near the top

        # Write the emotion text at the top center
        cv2.putText(frame, smoothed_emotion, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_4)

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Face Detection and Analysis', frame)

        # Check for the Esc key (27) to break the loop
        if cv2.waitKey(1) & 0xFF == 27:
            print("Esc key pressed. Exiting loop...")
            break
finally:
    # Release the capture and writer objects
    print("Releasing resources...")
    cam.release()
    out.release()
    cv2.destroyAllWindows()


# Face detect: Neutral, Suprise, Happy, Sad, disgusting, fear, angry 