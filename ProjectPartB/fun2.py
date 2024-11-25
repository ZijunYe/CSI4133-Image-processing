# Import the required modules
import cv2
from deepface import DeepFace

# File path to the input image
input_image_path = 'fun2.jpg'  # Change to your image path

# Read the input image
image = cv2.imread(input_image_path)
if image is None:
    print(f"Failed to read the image from {input_image_path}. Please check the file path.")
    exit()

# Haar Cascade File Path
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the frame to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
faces = facecascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("No faces detected in the image.")
else:
    print(f"Detected {len(faces)} face(s) in the image.")

# Process each face detected
for idx, (x, y, w, h) in enumerate(faces):
    # Crop the face region for individual analysis
    face_roi = image[y:y + h, x:x + w]

    # Analyze the cropped face region using DeepFace
    result = DeepFace.analyze(face_roi, actions=['emotion', 'race', 'gender', 'age'], enforce_detection=False)

    # Extract information from analysis
    emotion = result[0]['dominant_emotion']
    race = result[0]['dominant_race']
    age = str(result[0]['age'])  # Convert age to string
    gender = result[0]['dominant_gender']

    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Adjust the text position above the face rectangle
    text_position = y - 20 if y - 20 > 20 else y + h + 20

    # Increase font scale and adjust position for each text line
    font_scale = 0.8
    line_thickness = 2

    # Annotate the image with the analysis results
    cv2.putText(image, f"Face {idx + 1}", (x, text_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)
    cv2.putText(image, f"Emotion: {emotion}", (x, text_position + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
    cv2.putText(image, f"Race: {race}", (x, text_position + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
    cv2.putText(image, f"Gender: {gender}", (x, text_position + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)
    cv2.putText(image, f"Age: {age}", (x, text_position + 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness)

# Save the processed image
output_image_path = 'processed_image.jpg'
cv2.imwrite(output_image_path, image)
print(f"Processed image saved to {output_image_path}")

# Display the processed image
cv2.imshow('Face Detection and Analysis', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
