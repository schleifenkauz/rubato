import time
from collections import deque
from pythonosc import udp_client
import matplotlib.pyplot as plt
from util import compute_velocity, compute_acceleration, calculate_magnitude, avg_magnitude, update_lpf
from media import analyze_video

osc_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# Constants
alpha1 = 0.8
alpha2 = 0.25
alpha3 = 0.01
min_interval = 0.4

maxlen = 3
pos_history = deque(maxlen=maxlen)
vel_history = deque(maxlen=maxlen)
acc_history = deque(maxlen=maxlen)

# Lists for plotting
accel_lpf_list = []
threshold_list = []
velo_list = []
velo_threshold_list = []
peak_velo_list = []
beat_times = []
beats = []

# Global variables
velo_lpf = 0.25
last_beat_time = 0
peak_velo = 0
accel_lpf = 0
accel_threshold = 5

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
    global last_beat_time, peak_velo, accel_lpf, accel_threshold, velo_lpf
    acc_history.append(acc)
    avg_accel = avg_magnitude(acc_history)

    accel_lpf_before = accel_lpf
    accel_lpf = update_lpf(accel_lpf, avg_accel, alpha2)
    accel_threshold = update_lpf(accel_threshold, avg_accel, alpha3)

    avg_velo = avg_magnitude(vel_history)
    velo_threshold = max(velo_lpf / 1, 0.1)
    velo_lpf = update_lpf(velo_lpf, velo, alpha3)

    accel_lpf_list.append(accel_lpf)
    threshold_list.append(accel_threshold)
    velo_threshold_list.append(velo_lpf)
    peak_velo_list.append(peak_velo)

    if time.time() - last_beat_time < min_interval: return

    if avg_velo > peak_velo:  peak_velo = avg_velo

    if accel_lpf >= accel_threshold and calculate_magnitude(acc) < accel_lpf_before:
        if velo >= peak_velo / 3:
            print(f"{velo:.2f} > {peak_velo / 2.5:.2f}")
            return
        if velo >= velo_threshold:
            print(f"{velo:.2f} > {velo_threshold:.2f}")
            return
        peak_velo = 0
        detected_beat(accel_lpf, velo)



def detected_beat(magnitude, vel_magn):
    global last_beat_time
    beats.append(min(magnitude, 35))
    beat_times.append(len(peak_velo_list))
    print(f"Beat at {time.time():.2f}s: magnitude = {magnitude:.2f}, velo = {vel_magn:.2f}")
    osc_client.send_message("/beat", magnitude)
    last_beat_time = time.time()

def plot_data():
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(accel_lpf_list, label=f"Acceleration (alpha={alpha2})")
    plt.plot(threshold_list, label=f"Acceleration Threshold (alpha={alpha3})")
    plt.plot(velo_list, label='Velocity')
    plt.plot(peak_velo_list, label='Peak Velocity')
    plt.plot(velo_threshold_list, label='Velocity Threshold')
    # plt.scatter(beat_times, beats, label='Beats', color='black')
    for beat_time in beat_times:
        plt.axvline(x=beat_time, color='black', alpha=0.5, linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration Magnitude')
    plt.title('Acceleration Over Time')
    plt.ylim(0, 20)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyze_video(update_pos, show_video=True)
    plot_data()

