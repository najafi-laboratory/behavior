import soundfile as sf
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import scipy.signal

# analyzing the sound
audio_file_path = 'audio-recording-two.wav'
audio_data, sample_rate = sf.read(audio_file_path)
abs_audio_data = np.abs(audio_data)
peaks, _ = find_peaks(abs_audio_data, height=0.5)
time_ms = np.arange(len(audio_data)) / sample_rate * 1000
pulse_signal = np.zeros_like(audio_data)
pulse_signal[peaks] = 1

# figure to create the plot
time_vector = np.arange(len(audio_data)) / sample_rate
print(len(time_vector))
print(len(pulse_signal))
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

# note, need to find a different function to find the peaks as the threshhold is a rudimentary value right now.

