import cv2
import mediapipe as mp
import time
from collections import deque, namedtuple
from pythonosc import udp_client
import numpy as np
import matplotlib.pyplot as plt

def plot_acceleration():
    plt.figure(figsize=(20, 10))
    plt.plot(accel_list, label='Acceleration')
    plt.plot(velo_list, label='Velocity')
    plt.scatter(beat_times, beats, label='Beats', color= 'black')
    for beat_time in beat_times:
        plt.axvline(x=beat_time, color='black', alpha=0.5, linestyle='--')

    plt.axhline(y=beat_threshold, color='r', linestyle='--', label='Beat Threshold')
    plt.axhline(y=reset_threshold, color='g', linestyle='--', label='Reset Threshold')
    plt.ylim(0, 35)

    plt.xlabel('Sample')
    plt.ylabel('Acceleration Magnitude')
    plt.title('Acceleration Over Time')
    plt.legend()
    plt.show()


osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Add drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # Add drawing styles
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture('/home/nikolaus/cloud/rubato/dirigieren - cut.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

maxlen = 5
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen - 1)
acc_history = deque(maxlen=maxlen - 2)
time_history = deque(maxlen=maxlen)

accel_list = []
velo_list = []
beat_times = []
beats = []

last_beat_time = 0
min_interval = 0.4  # Minimum time between beats (in seconds)
MIN_TIME_DELTA = 0.001  # Minimum allowed time delta
last_peak = 0

def compute_velocity(pos1, pos2, time_delta):
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(p2 - p1) / time_delta for p1, p2 in zip(pos1, pos2)]


def compute_acceleration(vel1, vel2, time_delta):
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(v2 - v1) / time_delta for v1, v2 in zip(vel1, vel2)]


def calculate_magnitude(vector):
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5

def update_pos(pos):
    global last_beat_time
    if time_history and (time.time() - time_history[-1]) < MIN_TIME_DELTA: return
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
        velo_list.append(calculate_magnitude(vel))

        # Acceleration: second derivative of position
        if len(vel_history) >= 2 and len(time_deltas) >= 2:
            acc = compute_acceleration(vel_history[-2], vel_history[-1], time_deltas[-2])
            if len(acc_history) < 3:
                acc_history.append(acc)
                return
            update_acceleration(acc, calculate_magnitude(vel))


reset_threshold = 5 # 3 when using mp_hands
beat_threshold = 20 # 10 when using mp_hands
vel_threshold = 10

def update_acceleration(acc, velo):
    global last_beat_time, last_peak
    # avg_magn_before = sum(calculate_magnitude(accel) for accel in acc_history) / len(acc_history)
    acc_history.append(acc)
    avg_accel = sum(calculate_magnitude(accel) for accel in acc_history) / len(acc_history)
    magnitude = calculate_magnitude(acc)
    accel_list.append(min(magnitude, 50))
    if avg_accel <= reset_threshold:
        if last_peak != 0:
            last_peak = 0
            print("reset")
    if beat_threshold <= magnitude:
        if last_peak != 0:
            return
        if velo >= vel_threshold: return
        if time.time() - last_beat_time <= min_interval:
            return
        last_peak = magnitude
        detected_beat(magnitude, velo)


def calculate_average_point(points) -> list[float]:
    x_sum = sum(p.x for p in points)
    y_sum = sum(p.y for p in points)
    n = len(points)
    return [x_sum / n, y_sum / n]


def detected_beat(magnitude, vel_magn):
    global last_beat_time
    beats.append(min(magnitude, 35))
    beat_times.append(len(accel_list))
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}, velo = {vel_magn:.2f}")
    osc_client.send_message("/beat", magnitude)
    last_beat_time = time.time()


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros(image.shape, dtype=np.uint8)

    # if results.multi_hand_landmarks and results.multi_handedness:
    #     for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
    #         if results.multi_handedness[i].classification[0].label == "Left":
    #             mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #
    #             avg_point = calculate_average_point(hand_landmarks)
    #             update_pos(avg_point)
    # else :
    #     print("No hands detected")
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_hand = calculate_average_point(landmarks[16:22:2])
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = image.shape
        px = int(right_hand[0] * w)
        py = int(right_hand[1] * h)
        # Draw circle at right hand position
        cv2.circle(overlay, (px, py), 10, (0, 255, 0), -1)

        update_pos(right_hand)

    image = cv2.addWeighted(overlay, 1, image, 0.1, 0)
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plot_acceleration()