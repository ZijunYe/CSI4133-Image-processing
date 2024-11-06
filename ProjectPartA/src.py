
import cv2
import mediapipe as mp
import numpy as np

#1. Pre-set up the module and load the video source 
# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Specify the path to your video 
vidpath = 'videos/videos_partA.mp4'

# Initialize video capture
vidcap = cv2.VideoCapture(vidpath)


# Get information of the video 
fps = vidcap.get(cv2.CAP_PROP_FPS)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
# get frame sizes 
frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of frames
frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of frames

# Define output video path in the same folder
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #define the video compressed format 
result = cv2.VideoWriter('result.mp4',fourcc,fps,(frame_width,frame_height), isColor=True)

# Store hand centers to draw the movement path(tracking all the movements)
hand_paths = {'Left': [], 'Right': []} 


# 2. Initialize hand tracking
# create a instance of hands from class mediapipe.solution.hand 
# static_image_mode = false--> either track the hand independently in every frame, or live track the existing hnds 
# max-num_hand --> max number of hand in the video 
# min-detection_confidence = 0.5 --> threshold range between 0 to 1, presence of he hand, detection is valid 
with mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        # openCv usually uses BGR as default color  
        # MediaPipe usually uses RGB color format, therefore we need convert them first 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection/tracking
        processFrames = hands.process(rgb_frame)

        # Draw Green Rectangle around detected hands 
        if processFrames.multi_hand_landmarks and processFrames.multi_handedness:
           for hand_landmarks, hand_info in zip(processFrames.multi_hand_landmarks, processFrames.multi_handedness):
                # Get handedness (left or right) from the handedness object
                hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
                # print(f"Detected {hand_label} hand.") 

                # finding the max and min value between the hand landmarks
                # the value between max and min will be the height and width of the green rectangle 
                # that enable to draw the rectangle that fully covered the entire hand 
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                # for the movements
                sum_x, sum_y = 0, 0
                num_points = len(hand_landmarks.landmark)

                for landmark in hand_landmarks.landmark:
                    # landmark.x and y is the float number between 0 and 1, which determine where 
                    # frame.shape[1] is the width of the frame in pixels.
                    # frame.shape[0] is the height of the frame in pixels.
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                    # find the edge for the green rectangle 
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                    # sum all the points that present in the hand 
                    sum_x += x
                    sum_y += y
                
                # calculate the center of the hand by doing the mean value 
                center_x = int(sum_x / num_points)
                center_y = int(sum_y / num_points)

                
                hand_paths[hand_label].append((center_x, center_y))
                # print(f"Path length for {hand_label} hand: {len(hand_paths[hand_label])}")

                #draw the green rectangle around hand 
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)

                #trace the movement of the center of hand 
                path_color = (128, 0, 128) if hand_label == 'Left' else (255, 0, 0 )
                cv2.polylines(frame, [np.array(hand_paths[hand_label], np.int32)], isClosed=False, color=path_color, thickness=2)

                cv2.putText(frame, f"{hand_label} Hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color, 1, cv2.LINE_AA)
                
                
        # Write the frame with rectangles to the output video
        result.write(frame)

        # Display the resized frame
        cv2.imshow('Hand Tracking', frame)

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
vidcap.release()
result.release()
cv2.destroyAllWindows()