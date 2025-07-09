import cv2
import mediapipe as mp
import pyautogui
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbcontrol

class Gesture:
    PALM = "PALM"
    FIST = "FIST"
    V_GESTURE = "V_GESTURE"
    PINCH_MAJOR = "PINCH_MAJOR"
    PINCH_MINOR = "PINCH_MINOR"

class HandGestureControl:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.drawing_utils = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.volume_control = self.init_volume_control()
        self.last_volume = 0.5
        self.last_brightness = 50

    def init_volume_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def smooth_change(self, last_value, current_value, factor=0.3):
        return last_value * (1 - factor) + current_value * factor

    def get_gesture(self, landmarks):
        fingers = [
            landmarks.landmark[4].x < landmarks.landmark[3].x,  # Thumb
            landmarks.landmark[8].y < landmarks.landmark[6].y,  # Index
            landmarks.landmark[12].y < landmarks.landmark[10].y,  # Middle
            landmarks.landmark[16].y < landmarks.landmark[14].y,  # Ring
            landmarks.landmark[20].y < landmarks.landmark[18].y,  # Pinky
        ]

        if all(fingers):
            return Gesture.PALM
        elif not any(fingers):
            return Gesture.FIST
        elif fingers[1] and fingers[2] and not fingers[0]:
            return Gesture.V_GESTURE
        elif fingers[0] and fingers[1]:
            return Gesture.PINCH_MAJOR
        elif fingers[0] and fingers[2]:
            return Gesture.PINCH_MINOR

    def move_cursor(self, landmarks):
        index_tip = landmarks.landmark[8]
        cursor_x = int(index_tip.x * self.screen_width)
        cursor_y = int(index_tip.y * self.screen_height)
        pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

    def adjust_volume(self, landmarks):
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

        normalized_distance = min(max(distance, 0.02), 0.1)
        volume_level = (normalized_distance - 0.02) / (0.1 - 0.02)
        smoothed_volume = self.smooth_change(self.last_volume, volume_level)
        self.last_volume = smoothed_volume
        self.volume_control.SetMasterVolumeLevelScalar(smoothed_volume, None)

    def adjust_brightness(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[4]
            middle_tip = landmarks.landmark[12]
            distance = math.sqrt((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2)

            normalized_distance = min(max(distance, 0.02), 0.1)
            brightness_level = (normalized_distance - 0.02) / (0.1 - 0.02) * 100
            smoothed_brightness = self.smooth_change(self.last_brightness, brightness_level)
            self.last_brightness = smoothed_brightness
            sbcontrol.set_brightness(int(smoothed_brightness))
            print(f"Brightness adjusted to: {int(smoothed_brightness)}%")  # Debugging log

        except Exception as e:
            print(f"Error adjusting brightness: {e}")  # Debugging log

    def process_hand(self, landmarks, hand_label):
        gesture = self.get_gesture(landmarks)

        if gesture == Gesture.V_GESTURE:
            self.move_cursor(landmarks)
        elif gesture == Gesture.PINCH_MAJOR and hand_label == "Right":
            self.adjust_volume(landmarks)
        elif gesture == Gesture.PINCH_MINOR and hand_label == "Left":
            self.adjust_brightness(landmarks)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                results = self.process_frame(frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
                        self.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                        )
                        hand_label = hand_class.classification[0].label
                        self.process_hand(hand_landmarks, hand_label)

                cv2.imshow('Hand Gesture Control', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureControl()
    controller.run()
