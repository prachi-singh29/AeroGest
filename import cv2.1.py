import cv2
import mediapipe as mp
import pyautogui
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size for cursor control
screen_width, screen_height = pyautogui.size()

# Setup for volume control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Start webcam feed
cap = cv2.VideoCapture(0)

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def is_v_gesture(landmarks):
    """Detect if the V gesture is present by checking the index and middle finger tips."""
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    index_base = landmarks[5]
    middle_base = landmarks[9]

    if (index_tip.y < index_base.y) and (middle_tip.y < middle_base.y):
        if abs(index_tip.x - middle_tip.x) > 0.05:
            return True
    return False

def control_volume(landmarks):
    """Adjust volume based on the distance between the thumb and index finger of the right hand."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = distance(thumb_tip, index_tip)

    # Map the distance to volume level (0.0 to 1.0)
    volume_level = np.interp(dist, [0.02, 0.2], [0.0, 1.0])
    volume.SetMasterVolumeLevelScalar(volume_level, None)

def control_brightness(landmarks):
    """Adjust brightness based on the distance between the thumb and index finger of the left hand."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = distance(thumb_tip, index_tip)

    # Map the distance to brightness level (0 to 100)
    brightness_level = np.interp(dist, [0.02, 0.2], [0, 100])
    sbc.set_brightness(int(brightness_level))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a natural selfie view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Identify if the hand is left or right
            handedness = results.multi_handedness[0].classification[0].label

            # V gesture for cursor control
            if is_v_gesture(hand_landmarks.landmark):
                index_tip = hand_landmarks.landmark[8]
                screen_x = int(index_tip.x * screen_width)
                screen_y = int(index_tip.y * screen_height)
                pyautogui.moveTo(screen_x, screen_y)

                # Optional: Draw a circle at the index finger tip for visualization
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Volume control with right hand
            if handedness == 'Right':
                control_volume(hand_landmarks.landmark)

            # Brightness control with left hand
            if handedness == 'Left':
                control_brightness(hand_landmarks.landmark)

    # Display the frame
    cv2.imshow("Hand Gesture Control", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
