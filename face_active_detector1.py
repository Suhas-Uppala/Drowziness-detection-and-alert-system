import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import os

# Optional: For headless environments, force pygame to use a dummy audio driver
# os.environ["SDL_AUDIODRIVER"] = "dummy"

# Initialize pygame mixer for sound alert
pygame.mixer.init()
pygame.mixer.music.load("mixkit-alert-alarm-1005.wav")  # Ensure alarm.wav is in the same directory

# Function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # eye is a list of 6 (x, y) tuples: p1, p2, p3, p4, p5, p6
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold and consecutive frame length for triggering alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For drawing facial landmarks
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Define eye landmark indices for MediaPipe Face Mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Create a copy of the frame for the normal face window
    normal_frame = frame.copy()

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    ear = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape  # image height, width, channels

            # Convert landmark coordinates to pixel positions
            landmarks = [(int(lm.x * iw), int(lm.y * ih)) for lm in face_landmarks.landmark]

            # Get left and right eye coordinates based on defined indices
            left_eye = [landmarks[idx] for idx in LEFT_EYE_IDX]
            right_eye = [landmarks[idx] for idx in RIGHT_EYE_IDX]

            # Draw eye landmarks on the detection frame
            for point in left_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Calculate EAR for both eyes
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Display EAR on the detection frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if the EAR is below the blink threshold and increment counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        pygame.mixer.music.play(-1)  # Start alarm sound
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False
                    pygame.mixer.music.stop()

            # Optionally, draw the full face mesh on the detection frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    else:
        # If no face is detected, reset the counter and ensure the alarm is off.
        COUNTER = 0
        if ALARM_ON:
            ALARM_ON = False
            pygame.mixer.music.stop()

    # Display the two windows: Normal Face and Drowsiness Detection
    cv2.imshow("Normal Face", normal_frame)
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
