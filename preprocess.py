import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pickle
import torch
import pyaudio
from playsound import playsound
import sounddevice as sd


def find_frequency():
    sample_rate, signal = wav.read("C:/spins/vowels/m04iy.wav")
    plt.plot(signal)
    plt.show()
    # print(signal)
    print(sample_rate)
    sample_rate = sample_rate * 3e6
    print(f"dt={1/sample_rate} s")
    signal = signal[2500:8000]
    signal = signal / np.max(np.abs(signal))
    frequency_spectrum = np.fft.fft(signal)
    frequency = np.fft.fftfreq(len(signal), 1 / sample_rate)
    positive_frequencies = frequency[: len(frequency) // 2]
    positive_spectrum = np.abs(frequency_spectrum[: len(frequency) // 2])
    peak_frequency = positive_frequencies[np.argmax(positive_spectrum)]

    # Optional: Plotting the frequency spectrum for visualization
    average_frequency = np.sum(positive_frequencies * positive_spectrum) / np.sum(
        positive_spectrum
    )
    print(f"Average frequency: {average_frequency:.2f} Hz")
    plt.figure(figsize=(12, 6))
    plt.plot(positive_frequencies, positive_spectrum)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()
    print("peak_frequency: ")
    print(peak_frequency)


def play_sound(signal, sample_rate):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    stream.write(signal.astype(np.int16).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


def get_signal(filename: str):
    sample_rate, signal = wav.read(filename)
    playsound(filename)
    plt.plot(signal)
    plt.show()
    play_sound(signal[2000:8000], sample_rate)
    c = "y"
    f = "y"
    while True:
        try:
            c = input("Enter start sample index (or 'n' to stop): ")
            if c.lower() == "n":
                break
            f = input("Enter end sample index (or 'n' to stop): ")
            if f.lower() == "n":
                break

            start_idx = int(c)
            end_idx = int(f)
            if start_idx < 0 or end_idx > len(signal) or start_idx >= end_idx:
                print("Invalid indices. Please try again.")
                continue
            print(f"start_idx: {start_idx}, end_idx: {end_idx}")
            # Play the selected portion of the signal
            try:
                play_sound(signal[start_idx:end_idx], sample_rate=sample_rate)
            except Exception as e:
                print(f"Error playing sound: {e}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    if signal.shape[0] < 8000:
        signal = signal[1000:6000]
    else:
        signal = signal[2500:7500]
    signal = signal / np.max(np.abs(signal))  # normalize the waveform
    return signal


def extract_signal(filename: str, train: bool = True):
    print("extracting signal...")
    train_ranges = [4000, 3500, 4000, 3500, 3000, 3000, 3000, 3000, 3000]
    test_ranges = [2000, 2500, 2000, 2500, 2000, 2000, 1000, 1000, 1500]
    sample_rate, signal = wav.read(filename)
    if train:
        start = train_ranges.pop(0)
        signal = signal[start : start + 2000]
        plt.plot(signal)
        plt.title(filename)
        plt.show()
        print(signal.shape)
    else:
        start = test_ranges.pop(0)
        signal = signal[start : start + 2000]
        plt.plot(signal)
        plt.show()
    # assert train_ranges == [] and test_ranges == [], "Ranges not exhausted"
    signal = signal / np.max(np.abs(signal))
    return signal


def preprocess():
    files_to_extract = [
        "C:/spins/vowels/m01ei.wav",
        "C:/spins/vowels/m01ae.wav",
        "C:/spins/vowels/m01iy.wav",
        "C:/spins/vowels/m02ei.wav",
        "C:/spins/vowels/m02ae.wav",
        "C:/spins/vowels/m02iy.wav",
        "C:/spins/vowels/m03ei.wav",
        "C:/spins/vowels/m03ae.wav",
        "C:/spins/vowels/m03iy.wav",
    ]
    test_files_to_extract = [
        "C:/spins/vowels/m04ei.wav",
        "C:/spins/vowels/m04ae.wav",
        "C:/spins/vowels/m04iy.wav",
        "C:/spins/vowels/m07ei.wav",
        "C:/spins/vowels/m07ae.wav",
        "C:/spins/vowels/m07iy.wav",
        "C:/spins/vowels/m06ei.wav",
        "C:/spins/vowels/m06ae.wav",
        "C:/spins/vowels/m06iy.wav",
    ]
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    signals = [extract_signal(file, train=True) for file in files_to_extract]
    test_signals = [extract_signal(file, train=False) for file in test_files_to_extract]
    print(np.array(signals).shape)
    # for t in test_signals:
    #     print(len(t))
    print(np.array(test_signals).shape)
    with open(f"C:\spins\data\data.p", "wb") as data_file:
        pickle.dump(
            {
                "signals": torch.tensor(np.array(signals), dtype=torch.long),
                "test_signals": torch.tensor(np.array(test_signals), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            },
            data_file,
        )
    print(f'Data has been dumped into {"C:/spins/data"}/data.p!')


preprocess()
# find_frequency()
