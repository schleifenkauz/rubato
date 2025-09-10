import matplotlib.pyplot as plt

def plot_data(
        accel_lpf_list, threshold_list, velo_list, peak_velo_list, velo_threshold_list,
        tempo_list, beat_times,
        alpha1, alpha2,
):
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(accel_lpf_list, label=f"Acceleration (alpha={alpha1})")
    plt.plot(threshold_list, label=f"Acceleration Threshold (alpha={alpha2})")
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
    # plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(tempo_list, label='Tempo')
    plt.xlabel('Sample')
    plt.ylabel('Tempo')
    plt.title('Tempo Over Time')
    plt.ylim(0, 120)
    # plt.show()
