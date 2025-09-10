import numpy as np
import simpleaudio as sa

def beep(frequency=440, duration=0.1):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(frequency * 2 * np.pi * t) * 0.05
    audio = (wave * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, fs)