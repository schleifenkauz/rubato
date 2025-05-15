import cv2
import mediapipe as mp
import time
from collections import deque
from pythonosc import udp_client

import numpy as np

osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Add drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # Add drawing styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Video Capture ---
cap = cv2.VideoCapture('/home/nikolaus/cloud/rubato/dirigieren.mp4')

# --- Rolling Buffers ---
maxlen = 5
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen - 1)
acc_history = deque(maxlen=maxlen - 2)
time_history = deque(maxlen=maxlen)

last_beat_time = 0
min_interval = 0.4  # Minimum time between beats (in seconds)
MIN_TIME_DELTA = 0.001  # Minimum allowed time delta
last_peak = 0


# --- Helper Functions ---
def compute_velocity(pos1, pos2, time_delta):
    """Compute velocity (change in position over time)."""
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(p2 - p1) / time_delta for p1, p2 in zip(pos1, pos2)]


def compute_acceleration(vel1, vel2, time_delta):
    """Compute acceleration (change in velocity over time)."""
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(v2 - v1) / time_delta for v1, v2 in zip(vel1, vel2)]


def calculate_magnitude(vector):
    """Calculate the magnitude of a 2D vector."""
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5


def update_pos(pos):
    global last_beat_time
    time_history.append(time.time())
    pos_history.append(pos)

    if len(pos_history) == maxlen:
        # Convert deque to list for slicing
        time_list = list(time_history)
        # Calculate time deltas with safety check
        time_deltas = []
        for t1, t2 in zip(time_list[:-1], time_list[1:]):
            delta = t2 - t1
            time_deltas.append(max(delta, MIN_TIME_DELTA))

        # Velocity: first derivative of position
        vel = compute_velocity(pos_history[-2], pos_history[-1], time_deltas[-1])
        vel_history.append(vel)

        # Acceleration: second derivative of position
        if len(vel_history) >= 2 and len(time_deltas) >= 2:
            acc = compute_acceleration(vel_history[-2], vel_history[-1], time_deltas[-2])
            acc_magnitude = calculate_magnitude(acc)
            update_acceleration(acc_magnitude)
            acc_history.append(acc)


reset_threshold = 5
beat_threshold = 10


def update_acceleration(magnitude):
    global last_beat_time, last_peak
    if len(acc_history) < 3: return
    acc_norms = [calculate_magnitude(acc_history[j]) for j in range(3)]
    if magnitude <= reset_threshold:
        if last_peak != 0:
            last_peak = 0
            print("reset")
    elif magnitude >= beat_threshold:
        if acc_norms[1] <= acc_norms[0] or acc_norms[1] <= acc_norms[2]: return
        if last_peak != 0:
            print("beat threshold reached, but not reset")
            return
        if time.time() - last_beat_time <= min_interval:
            print("min interval not reached yet")
            return
        last_peak = magnitude
        detected_beat(magnitude)


def detected_beat(magnitude):
    global last_beat_time
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}")
    osc_client.send_message("/beat", magnitude)
    last_beat_time = time.time()


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros(image.shape, dtype=np.uint8)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness[i].classification[0].label == "Left":
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                idx_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if not time_history or (time.time() - time_history[-1]) >= MIN_TIME_DELTA:
                    update_pos([idx_finger_tip.x, idx_finger_tip.y])

    # image = cv2.addWeighted(overlay, 1, image,  0.5, 0)
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
