# Import the required modules
import cv2
from deepface import DeepFace
import numpy as np

# Store recent emotion predictions for smoothing
recent_predictions = []

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
out = cv2.VideoWriter('result.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Face Analysis using DeepFace
    result = DeepFace.analyze(frame, actions=['emotion', 'race', 'gender', 'age'], enforce_detection=False)

    # Extract information from analysis
    emotion = result[0]['dominant_emotion']
    race = result[0]['dominant_race']
    age = str(result[0]['age'])  # Convert age to string
    gender = result[0]['dominant_gender']

    # Smooth emotions by storing recent predictions
    recent_predictions.append(emotion)
    if len(recent_predictions) > 5:  # Keep the last 5 predictions
        recent_predictions.pop(0)
    smoothed_emotion = max(set(recent_predictions), key=recent_predictions.count)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
            # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Adjust the text position above the face rectangle
        text_position = y - 20 if y - 20 > 20 else y + h + 20

        # Increase font scale and adjust position for each text line
        font_scale = 1.0
        line_thickness = 2

        cv2.putText(frame, f"Emotion: {smoothed_emotion}", (x, text_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
        cv2.putText(frame, f"Race: {race}", (x, text_position + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
        cv2.putText(frame, f"Gender: {gender}", (x, text_position + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
        cv2.putText(frame, f"Age: {age}", (x, text_position + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
            
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
