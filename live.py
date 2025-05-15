import cv2
import mediapipe as mp
import time
from collections import deque
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from pythonosc import udp_client
import numpy as np
import matplotlib.pyplot as plt

def plot_acceleration():
    plt.figure(figsize=(20, 10))
    plt.plot(accel_lpf2_list, label='Average Acceleration')
    plt.plot(accel_lpf1_list, label='Acceleration')
    plt.plot(threshold_list, label='Threshold')
    plt.plot(velo_list, label='Velocity')
    plt.scatter(beat_times, beats, label='Beats', color='black')
    for beat_time in beat_times:
        plt.axvline(x=beat_time, color='black', alpha=0.5, linestyle='--')

    plt.ylim(0, 25)

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
        time_list = list(time_history)
        time_deltas = []
        for t1, t2 in zip(time_list[:-1], time_list[1:]):
            delta = t2 - t1
            time_deltas.append(max(delta, MIN_TIME_DELTA))

        vel = compute_velocity(pos_history[-2], pos_history[-1], time_deltas[-1])
        vel_history.append(vel)
        velo_list.append(calculate_magnitude(vel))

        if len(vel_history) >= 2 and len(time_deltas) >= 2:
            acc = compute_acceleration(vel_history[-2], vel_history[-1], time_deltas[-2])
            if len(acc_history) < 3:
                acc_history.append(acc)
                return
            update_acceleration(acc, calculate_magnitude(vel))


velo_lpf = 0.25
alpha1 = 0.8
alpha2 = 0.1
alpha3 = 0.01

maxlen = 5
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen)
acc_history = deque(maxlen=maxlen)
time_history = deque(maxlen=maxlen)

accel_lpf1_list = []
accel_lpf2_list = []
threshold_list = []
velo_list = []
beat_times = []
beats = []

last_beat_time = 0
min_interval = 0.4  # Minimum time between beats (in seconds)
MIN_TIME_DELTA = 0.001  # Minimum allowed time delta
last_peak = 0

accel_lpf1 = 0
accel_lpf2 = 0
threshold = 5

def update_lpf(now, value, alpha):
    return alpha * value + (1 - alpha) * now

def avg(source):
    return sum(calculate_magnitude(v) for v in source) / len(source)

def update_acceleration(acc, velo):
    global last_beat_time, last_peak, accel_lpf1, accel_lpf2, threshold, velo_lpf
    acc_history.append(acc)
    avg_accel = avg(acc_history)

    accel_lpf1 = update_lpf(accel_lpf1, avg_accel, alpha1)
    accel_lpf2_before = accel_lpf2
    accel_lpf2 = update_lpf(accel_lpf2, avg_accel, alpha2)
    threshold = update_lpf(threshold, avg_accel, alpha3)

    velo_threshold = velo_lpf / 3
    velo_lpf = update_lpf(velo_lpf, velo, alpha3)

    accel_lpf1_list.append(accel_lpf1)
    accel_lpf2_list.append(accel_lpf2)
    threshold_list.append(threshold)

    if accel_lpf1 * 1.5 <= threshold:
        if last_peak != 0:
            last_peak = 0
            print("reset")
    magnitude = calculate_magnitude(acc)
    if accel_lpf1 * 1.5 >= threshold and magnitude < accel_lpf2_before:
        if last_peak != 0:
            print("not reset yet")
            return
        if velo >= velo_threshold:
            print(f"{velo:.2f} > {velo_threshold:.2f}")
            return
        if time.time() - last_beat_time <= min_interval:
            return
        last_peak = 1
        detected_beat(avg_accel, velo)


def calculate_average_point(points) -> list[float]:
    x_sum = sum(p.x for p in points)
    y_sum = sum(p.y for p in points)
    n = len(points)
    return [x_sum / n, y_sum / n]


def detected_beat(magnitude, vel_magn):
    global last_beat_time
    beats.append(min(magnitude, 35))
    beat_times.append(len(accel_lpf1_list))
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}, velo = {vel_magn:.2f}")
    osc_client.send_message("/beat", magnitude)
    last_beat_time = time.time()


def draw_circle(target, point, radius, color):
    h, w, _ = target.shape
    px = int(point[0] * w)
    py = int(point[1] * h)

    cv2.circle(target, (px, py), radius, color, thickness=-1)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 360))
    results = pose.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros(image.shape, dtype=np.uint8)

    # if results.multi_hand_landmarks and results.multi_handedness:
    #     for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
    #         if results.multi_handedness[i].classification[0].label == "Left":
    #             mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #
    #             avg_point = calculate_average_point(hand_landmarks.landmark)
    #             update_pos(avg_point)
    # else :
    #     print("No hands detected")
    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        right_hand = calculate_average_point(landmarks.landmark[16:22:2])
        update_pos(right_hand)

        mp_drawing.draw_landmarks(overlay, landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec((20, 20, 20), -1, 3))
        draw_circle(overlay, right_hand, 5, color=(0, 255, 0))

    image = cv2.addWeighted(overlay, 1, image, 0.1, 0)
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plot_acceleration()
