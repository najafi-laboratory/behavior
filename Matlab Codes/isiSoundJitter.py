import soundfile as sf
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def audioAnalysis(file):
    # analyzing the sound through sound file to get the sample rate and audio data
    audio_file_path = file
    audio_data, sample_rate = sf.read(audio_file_path)
    abs_audio_data = np.abs(audio_data)
    peaks, _ = find_peaks(abs_audio_data, height=0.5)
    time_ms = np.arange(len(audio_data)) / sample_rate * 1000000
    pulse_signal = np.zeros_like(audio_data)
    pulse_signal[peaks] = 1

    # gets the time between each beep in ms
    differences = []
    for x in range(len(peaks)):
        peaks[x] = peaks[x] / (sample_rate / 1000)

    for x in range(len(peaks) - 1):
        if peaks[x + 1] - peaks[x] > 50:
            differences.append(peaks[x + 1] - peaks[x])

    # creates an array in which each index is a different trial
    currTrial = []
    audioTrials = []
    for value in differences:
        if value > 1500:
            if currTrial:
                audioTrials.append(currTrial)
            currTrial = []
        else:
            currTrial.append(value)
    if currTrial:
        audioTrials.append(currTrial)

    # find the average time between beeps for each trial
    averageTrials = []
    for trial in audioTrials:
        average = 0
        for value in range(4, len(trial)):
            if trial[value] > 99:
                average += trial[value]
        if len(trial) > 5:
            average = round(average / (len(trial) - 4), 2)
            averageTrials.append(average)

    # provide an array in which the trials are separated.
    def print_formatted_array(arr):
        print("[")
        for sublist in arr:
            print("   [", end="")
            print(", ".join(map(repr, sublist)), end="],\n")
        print("]")
    print(len(audioTrials))
    print("Time between beeps:")
    print_formatted_array(audioTrials)
    print("Average time between beeps per trial:", averageTrials)

    for outlier in averageTrials:
        if (100.0 < outlier < 105.0) | (900.0 < outlier < 905.0):
            continue
        else:
            print("Trial:", averageTrials.index(outlier), "was an outlier:", outlier)

    # figure to create the pulse signal plot
    for x in range(len(peaks)):
        peaks[x] = peaks[x] * (sample_rate / 1000)
    time_vector = np.arange(len(audio_data)) / sample_rate
    pulse_signal = np.zeros_like(audio_data)
    pulse_signal[peaks] = 1
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, pulse_signal, label='Pulse Signal', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Pulse Signal from Detected Peaks')
    plt.legend()
    plt.grid()
    plt.show()

# code to run the different recordings
audioAnalysis('hard.wav')

