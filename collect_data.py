import os
import time
from datetime import datetime

import cv2
import mido

from media import analyze_video
from utility import beep

wrist_pos = []
beats = []
video_latency = 0.1

start_time = 0

global device, timestamp, filename, cap

def analyze_frame(right_hand, left_hand, fps):
    if start_time == 0: return
    wrist_pos.append(right_hand)

def select_input_device():
    names = mido.get_input_names()
    xjam = [name for name in names if "Xjam" in name]
    if len(xjam) == 1: return xjam[0]
    for i, name in enumerate(names):
        print(f"{i}: {name}")
    try:
        idx = int(input("Select input device: "))
        return names[idx]
    except ValueError:
        return None

def handle_midi(msg):
    global start_time
    if msg.type == "note_on" and msg.velocity > 0:
        if start_time == 0:
            start_time = time.time()
        if msg.note == 48:
            cap.release()
        else:
            t = time.time() - start_time
            beats.append(t)
            beep()
            print(f"Beat detected at {t:.2f}s. Note: {msg.note}", flush=True)

def finish():
    device.close()
    save_video = False
    save = input("Save data? (y/n): ")
    if save == "y":
        save_data()
        save_video = input("Save video? (y/n): ") == "y"
    if not save_video:
        os.remove(filename)


def save_data():
    features_file = f"data/{timestamp}-features.csv"
    with open(features_file, "w") as f:
        for pos in wrist_pos:
            f.write(f"{pos[0]:.3f},{pos[1]:.3f}\n")
        print(f"Saved wrist positions to {features_file}")

    beats_file = f"data/{timestamp}-beats.txt"
    with open(beats_file, "w") as f:
        for beat in beats:
            f.write(f"{beat:.3f}\n")
        print(f"Saved beats to {beats_file}")
    
def collect_data():
    global device, timestamp, filename, cap
    cap = cv2.VideoCapture(0)
    device_name = select_input_device()

    if device_name is None:
        print("No input device selected. Exiting.")
        return

    device = mido.open_input(device_name, callback=handle_midi)

    timestamp = datetime.now().strftime("%m-%d_%H_%M")
    filename = f"data/{timestamp}-video.avi"
    analyze_video(cap, "pose", analyze_frame, output_filename=filename)

    finish()

if __name__ == "__main__":
    collect_data()