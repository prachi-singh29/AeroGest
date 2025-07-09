import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbcontrol

# Gesture Enumeration
class Gest(IntEnum):
    PALM = 0
    FIST = 1
    PINCH_MAJOR = 2
    PINCH_MINOR = 3
    V_GEST = 4

# Hand Label Enumeration
class HLabel(IntEnum):
    MAJOR = 0
    MINOR = 1

# Hand Recognition Class
class HandRecog:
    def __init__(self, hand_label):
        self.hand_label = hand_label
        self.finger_states = [False] * 5
        self.gesture = None

    def update_hand_result(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks

    def set_finger_state(self):
        self.finger_states[0] = self.hand_landmarks.landmark[4].x < self.hand_landmarks.landmark[3].x
        for i in range(1, 5):
            self.finger_states[i] = (
                self.hand_landmarks.landmark[i * 4].y < self.hand_landmarks.landmark[i * 4 - 2].y
            )

    def get_gesture(self):
        if all(self.finger_states):
            self.gesture = Gest.PALM
        elif not any(self.finger_states):
            self.gesture = Gest.FIST
        elif self.finger_states[0] and self.finger_states[1]:
            self.gesture = Gest.PINCH_MAJOR
        elif self.finger_states[2] and not self.finger_states[0]:
            self.gesture = Gest.PINCH_MINOR
        elif self.finger_states[1] and self.finger_states[2]:
            self.gesture = Gest.V_GEST
        return self.gesture

# Controller Class for Volume, Brightness, Cursor Control, and Clicks
class Controller:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.volume_control = self.init_volume_control()
        self.last_volume = 0.5  # Initial volume (50%)
        self.last_brightness = None

    def init_volume_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def smooth_change(self, last_value, current_value, smoothing_factor=0.3):
        if last_value is None:
            return current_value
        return last_value * (1 - smoothing_factor) + current_value * smoothing_factor

    def change_volume(self, gesture, landmarks):
        if gesture == Gest.PINCH_MAJOR:
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            # Normalize distance and constrain
            normalized_distance = min(max(distance, 0.02), 0.1)
            volume_level = (normalized_distance - 0.02) / (0.1 - 0.02)  # Scaled to 0-1 range

            # Smooth transition
            smoothed_volume = self.smooth_change(self.last_volume, volume_level)
            self.last_volume = smoothed_volume

            # Set volume
            print(f"Setting volume to: {smoothed_volume:.2f}")
            self.volume_control.SetMasterVolumeLevelScalar(smoothed_volume, None)

    def change_brightness(self, gesture, landmarks):
        if gesture == Gest.PINCH_MINOR:
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            # Normalize distance
            normalized_distance = min(max(distance, 0.02), 0.1)
            brightness_level = (normalized_distance - 0.02) / (0.1 - 0.02) * 100

            # Smooth transition
            smoothed_brightness = self.smooth_change(self.last_brightness, brightness_level)
            self.last_brightness = smoothed_brightness

            print(f"Setting brightness to: {int(smoothed_brightness)}%")
            sbcontrol.set_brightness(int(smoothed_brightness))

    def move_cursor(self, gesture, landmarks):
        if gesture == Gest.V_GEST:
            index_tip = landmarks.landmark[8]
            cursor_x = int(index_tip.x * self.screen_width)
            cursor_y = int(index_tip.y * self.screen_height)
            print(f"Moving cursor to: ({cursor_x}, {cursor_y})")
            pyautogui.moveTo(cursor_x, cursor_y, duration=0)

    def click(self, gesture):
        if gesture == Gest.PALM:
            print("Clicking mouse.")
            pyautogui.click()

# Main Function
def main():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    controller = Controller()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    hand_recog = HandRecog(HLabel.MAJOR)
                    hand_recog.update_hand_result(hand_landmarks)
                    hand_recog.set_finger_state()
                    gesture = hand_recog.get_gesture()
                    print(f"Detected gesture: {gesture}")

                    controller.move_cursor(gesture, hand_landmarks)
                    controller.change_volume(gesture, hand_landmarks)
                    controller.change_brightness(gesture, hand_landmarks)
                    controller.click(gesture)

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

# Entry Point
if __name__ == "__main__":
    main()
