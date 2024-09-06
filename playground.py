import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
# sampling_rate = 100  # Hz
# duration = 2  # seconds
# frequency = 3  # Hz

# # Generate time axis
# t = torch.linspace(0, duration, int(sampling_rate * duration))

# # Create a sine wave
# signal = 4 * torch.sin(2 * np.pi * frequency * t)

# # Compute FFT
# fft_result = abs(torch.fft.fft(signal))
# fft_result[fft_result < 10] = 0
# print(fft_result)
# # Calculate frequency axis
# freq = torch.fft.fftfreq(len(signal), 1 / sampling_rate)
# print("freq: ")
# print(freq)
# mult = freq * fft_result
# average = mult[: mult.shape[0] // 2].sum() / abs(fft_result[: mult.shape[0] // 2]).sum()
# print(f"average: {average}")
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(t, signal)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Sine Wave")

# # Plot the FFT magnitude
# plt.subplot(1, 2, 2)
# plt.plot(freq, abs(fft_result))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title("FFT Magnitude")
# plt.xlim(0, 5)  # Zoom in on the positive frequency axis

# plt.tight_layout()
# plt.show()


# def extract_average_frequency(signal: torch.tensor):
#     fft_result = abs(torch.fft.fft(signal))
#     fft_result[fft_result < 10] = 0
#     freq = torch.fft.fftfreq(len(signal), 1 / sampling_rate)
#     mult = freq * fft_result
#     average = (
#         mult[: mult.shape[0] // 2].sum() / abs(fft_result[: mult.shape[0] // 2]).sum()
#     )
#     return average


def show_plot():
    to_plot = torch.tensor(
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    plt.figure(figsize=(12, 4))
    plt.plot(to_plot)
    plt.xlabel("time(t)")
    plt.ylabel("Intensity(mT)")
    plt.show()


show_plot()
