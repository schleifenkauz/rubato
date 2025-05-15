import time
from collections import deque
from pythonosc import udp_client
import matplotlib.pyplot as plt
from util import compute_velocity, compute_acceleration, calculate_magnitude, avg_magnitude, update_lpf
from media import analyze_video

osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

velo_lpf = 0.25
alpha1 = 0.8
alpha2 = 0.2
alpha3 = 0.03

maxlen = 3
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen)
acc_history = deque(maxlen=maxlen)

accel_lpf1_list = []
accel_lpf2_list = []
threshold_list = []
velo_list = []
beat_times = []
beats = []

last_beat_time = 0
min_interval = 0.4  # Minimum time between beats (in seconds)
peak_velo = 0

accel_lpf1 = 0
accel_lpf2 = 0
accel_lpf3 = 5

def update_pos(pos, fps):
    pos_history.append(pos)

    if len(pos_history) == maxlen:
        vel = compute_velocity(pos_history[-2], pos_history[-1], 1 / fps)
        vel_history.append(vel)
        velo_list.append(calculate_magnitude(vel))

        if len(vel_history) >= 2:
            acc = compute_acceleration(vel_history[-2], vel_history[-1], 1 / fps)
            if len(acc_history) < 3:
                acc_history.append(acc)
                return
            update_acceleration(acc, calculate_magnitude(vel))

def update_acceleration(acc, velo):
    global last_beat_time, peak_velo, accel_lpf1, accel_lpf2, accel_lpf3, velo_lpf
    acc_history.append(acc)
    avg_accel = avg_magnitude(acc_history)

    accel_lpf1 = update_lpf(accel_lpf1, avg_accel, alpha1)
    accel_lpf2_before = accel_lpf2
    accel_lpf2 = update_lpf(accel_lpf2, avg_accel, alpha2)
    accel_threshold = min(accel_lpf2, 3)
    accel_lpf3 = update_lpf(accel_lpf3, avg_accel, alpha3)

    avg_velo = avg_magnitude(vel_history)

    velo_threshold = max(velo_lpf / 1, 0.1)
    velo_lpf = update_lpf(velo_lpf, velo, alpha3)

    accel_lpf1_list.append(accel_lpf1)
    accel_lpf2_list.append(accel_lpf2)
    threshold_list.append(accel_lpf3)

    if time.time() - last_beat_time < min_interval: return

    magnitude = calculate_magnitude(acc)
    if avg_velo > peak_velo and avg_velo > velo_threshold:
        peak_velo = avg_velo
        print(f"peak_velo = {peak_velo}")
    if accel_lpf1 >= accel_threshold * 1.5 and magnitude < accel_lpf2_before:
        if velo > peak_velo / 2.5:
            print(f"{velo:.2f} > {peak_velo / 2.5:.2f}")
            return
        if velo >= velo_threshold:
            print(f"{velo:.2f} > {velo_threshold:.2f}")
            return
        peak_velo = 0
        detected_beat(avg_accel, velo)



def detected_beat(magnitude, vel_magn):
    global last_beat_time
    beats.append(min(magnitude, 35))
    beat_times.append(len(accel_lpf1_list))
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}, velo = {vel_magn:.2f}")
    osc_client.send_message("/beat", magnitude)
    last_beat_time = time.time()

def plot_data():
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(accel_lpf2_list, label='Average Acceleration')
    plt.plot(accel_lpf1_list, label='Acceleration')
    plt.plot(threshold_list, label='Threshold')
    plt.plot(velo_list, label='Velocity')
    plt.scatter(beat_times, beats, label='Beats', color='black')
    for beat_time in beat_times:
        plt.axvline(x=beat_time, color='black', alpha=0.5, linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration Magnitude')
    plt.title('Acceleration Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyze_video(update_pos, show_video=True)
    plot_data()

