import cv2
from deepface import DeepFace

# Initialize the Haar Cascade for face detection
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facecascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (face)
        roi = frame[y:y+h, x:x+w]

        try:
            # Analyze the face
            analysis = DeepFace.analyze(roi, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
            dominant_emotion = analysis['dominant_emotion']
            age = int(analysis['age'])
            gender = analysis['gender']

            # Overlay results on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error during DeepFace analysis: {e}")
    
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection and Analysis", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()




# Face detect: Neutral, Fear, Suprise, Happy, Sad 