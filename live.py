import time
from collections import deque
from pythonosc import udp_client
import matplotlib.pyplot as plt
from util import *
from media import analyze_video
from datetime import datetime

osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# Constants
ALPHA1 = 0.25
ALPHA2 = 0.03
TEMPO_ALPHA = 0.35
MIN_CONFIDENCE = 0.9
CONFIDENCE_MULT = 10
MIN_INTERVAL = 0.4
PEAK_FACTOR = 0.25

maxlen = 3
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen)
acc_history = deque(maxlen=maxlen)

# Lists for plotting
accel_lpf_list = []
threshold_list = []
velo_list = []
velo_threshold_list = []
confidence_list = []
peak_velo_list = []
beat_times = []
beats = []
tempo_list = []

# Global variables
velo_lpf = 0.25
last_beat_time = 0
peak_velo = 0
accel_lpf = 0
accel_threshold = 5
magnitude_lpf = 0.1
expected_interval = 1
average_interval = 1

def update_pos(right_hand, left_hand, fps):
    pos_history.append(right_hand)

    if len(pos_history) == maxlen:
        vel = compute_velocity(pos_history[-2], pos_history[-1], 1 / fps)
        vel_history.append(vel)
        velo_list.append(calculate_magnitude(vel))

        if len(vel_history) >= 2:
            acc = compute_acceleration(vel_history[-2], vel_history[-1], 1 / fps)
            if len(acc_history) < 3:
                acc_history.append(acc)
                return
            acc_magnitude = (calculate_magnitude(vel_history[-1]) - calculate_magnitude(vel_history[-2]))
            #avg_velo = calculate_average_vector(vel_history)
            update_acceleration(acc, acc_magnitude, calculate_magnitude(vel))
    
    left_hand[1] = 1 - left_hand[1]
    osc_client.send_message("/left_hand", left_hand)

velo_lpf2 = 0.2

def update_acceleration(acc, magnitude_acc, velo):
    global last_beat_time, peak_velo, accel_lpf, accel_threshold, velo_lpf, average_interval, expected_interval, magnitude_lpf, velo_lpf2
    acc_history.append(acc)
    avg_accel = avg_magnitude(acc_history)

    accel_lpf = update_lpf(accel_lpf, avg_accel, ALPHA1)
    accel_threshold = update_lpf(accel_threshold, avg_accel, ALPHA2)
    magnitude_lpf = update_lpf(magnitude_lpf, magnitude_acc, 0.12)

    avg_velo = avg_magnitude(vel_history)
    max_velo = min(velo_lpf, peak_velo * PEAK_FACTOR)
    velo_lpf = update_lpf(velo_lpf, velo, ALPHA2)
    velo_lpf2 = update_lpf(velo_lpf2, velo, 0.5)

    accel_lpf_list.append(accel_lpf)
    threshold_list.append(accel_threshold)
    velo_threshold_list.append(max_velo)
    peak_velo_list.append(peak_velo)
    tempo_list.append(60 / average_interval)

    current_time = time.time()
    interval = current_time - last_beat_time
    if interval < MIN_INTERVAL:
        confidence_list.append(0)
        return

    if avg_velo > peak_velo: peak_velo = avg_velo

    confidence = 1
    confidence *= asymmetric_sigmoid(interval / average_interval, k1=0.5, k2=0.25)
    confidence *= asymmetric_sigmoid(accel_lpf / accel_threshold, k1=3, k2=1)
    confidence *= asymmetric_sigmoid(1 - magnitude_lpf, k1=2, k2=8)
    confidence *= asymmetric_sigmoid(max_velo / velo_lpf2, k1=2, k2=3)
    confidence_list.append(confidence * CONFIDENCE_MULT)

    if confidence >= MIN_CONFIDENCE:
        detected_beat(accel_lpf, velo_lpf2)
        last_beat_time = current_time
        average_interval = update_lpf(average_interval, interval, TEMPO_ALPHA)
        peak_velo = 0


def detected_beat(magnitude, vel_magn):
    beats.append(min(magnitude, 35))
    beat_times.append(len(peak_velo_list))
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}, velo = {vel_magn:.2f}")
    osc_client.send_message("/beat", magnitude)


def plot_data():
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(accel_lpf_list, label=f"Acceleration (alpha={ALPHA1})")
    plt.plot(threshold_list, label=f"Acceleration Threshold (alpha={ALPHA2})")
    plt.plot(velo_list, label='Velocity')
    plt.plot(peak_velo_list, label='Peak Velocity')
    plt.plot(velo_threshold_list, label='Velocity Threshold')
    # plt.plot(confidence_list, label='Confidence', scaley=False)
    # plt.axhline(y=MIN_CONFIDENCE * CONFIDENCE_MULT, color='red', linestyle='--', label='Minimum Confidence')
    # plt.scatter(beat_times, beats, label='Beats', color='black')
    for beat_time in beat_times:
        plt.axvline(x=beat_time, color='black', alpha=0.5, linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration Magnitude')
    plt.title('Acceleration Over Time')
    plt.ylim(0, 20)
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(tempo_list, label='Tempo')
    plt.xlabel('Sample')
    plt.ylabel('Tempo')
    plt.title('Tempo Over Time')
    plt.ylim(0, 120)
    # plt.show()


if __name__ == "__main__":
    last_beat_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    analyze_video(update_pos, show_video=True, output_filename=f"data/video{timestamp}.avi")
    plot_data()
