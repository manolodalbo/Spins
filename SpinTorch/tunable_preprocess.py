import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pickle
import torch
import pyaudio
from playsound import playsound
import sounddevice as sd
import os.path


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


def extract_middle_signal(filename: str, middle_size: int):
    sample_rate, signal = wav.read(filename)
    signal = signal[len(signal) // 2 - middle_size : len(signal) // 2 + middle_size]
    signal = signal / np.max(np.abs(signal))
    return signal


def preprocess(middle_size=2000):
    files_to_extract = []
    train_labels = []
    for i in range(1, 26):
        if os.path.isfile(f"C:/spins/vowels/m{i:02d}ei.wav"):
            files_to_extract.append(f"C:/spins/vowels/m{i:02d}ei.wav")
            files_to_extract.append(f"C:/spins/vowels/m{i:02d}ae.wav")
            files_to_extract.append(f"C:/spins/vowels/m{i:02d}iy.wav")
            train_labels += [0, 1, 2]
        else:
            print(f"File m{i:02d} not found")
    test_labels = []
    test_files_to_extract = []
    for i in range(25, 49):
        if os.path.isfile(f"C:/spins/vowels/m{i:02d}ei.wav"):
            test_files_to_extract.append(f"C:/spins/vowels/m{i:02d}ei.wav")
            test_files_to_extract.append(f"C:/spins/vowels/m{i:02d}ae.wav")
            test_files_to_extract.append(f"C:/spins/vowels/m{i:02d}iy.wav")
            test_labels += [0, 1, 2]
        else:
            print(f"File m{i:02d}.wav not found")
    signals = [extract_middle_signal(file, middle_size) for file in files_to_extract]
    test_signals = [
        extract_middle_signal(file, middle_size) for file in test_files_to_extract
    ]
    data_dict = {
        "signals": torch.tensor(np.array(signals), dtype=torch.long),
        "test_signals": torch.tensor(np.array(test_signals), dtype=torch.long),
        "train_labels": torch.tensor(train_labels, dtype=torch.long),
        "test_labels": torch.tensor(test_labels, dtype=torch.long),
    }
    return data_dict
