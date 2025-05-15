import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands

# Offline video processing
cap = cv2.VideoCapture("/home/nikolaus/cloud/rubato/dirigieren.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)  # Useful for computing timestamps from frame index
frame_time = 1 / fps if fps > 0 else 1/25

timestamps = []
positions = []

with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    frame_idx = 0
    while True:
        success, image = cap.read()
        if not success:
            break

        if frame_idx % 1000 == 0:
            print(f"Processed {frame_time * frame_idx:.2f} seconds of video.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (640, 480))
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, handedness in enumerate(results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    hand = results.multi_hand_landmarks[i]
                    lm = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = lm.x, lm.y
                    t = frame_idx * frame_time

                    timestamps.append(t)
                    positions.append((x, y))
        frame_idx += 1

cap.release()

# Convert to arrays
timestamps = np.array(timestamps)
positions = np.array(positions)

# Ensure we have enough points
if len(positions) < 5:
    print("Not enough data to compute acceleration.")
    exit()

# Compute velocities (dx/dt, dy/dt)
dt = np.diff(timestamps)
dx = np.diff(positions[:, 0])
dy = np.diff(positions[:, 1])
vx = dx / dt
vy = dy / dt

# Compute acceleration (dv/dt)
dvx = np.diff(vx)
dvy = np.diff(vy)
dt2 = dt[1:]
ax = dvx / dt2
ay = dvy / dt2

# Compute acceleration magnitude
acc_magnitude = np.sqrt(ax**2 + ay**2)

# Find peaks in acceleration
peaks, _ = find_peaks(acc_magnitude, height=np.percentile(acc_magnitude, 65), distance=5)

# Estimate BPM
if len(peaks) >= 2:
    peak_times = timestamps[2:][peaks]  # skip first 2 because of diff
    intervals = np.diff(peak_times)
    bpm = 60 / np.mean(intervals)
    print(f"Estimated tempo: {bpm:.2f} BPM")
else:
    print("Not enough peaks to estimate tempo.")

# Plot acceleration + peaks
plt.plot(acc_magnitude)
plt.plot(peaks, acc_magnitude[peaks], "rx")
plt.title("Acceleration Peaks (Right Hand)")
plt.xlabel("Frame")
plt.ylabel("Acceleration Magnitude")
plt.show()
