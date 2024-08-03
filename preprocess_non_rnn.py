import numpy as np
import tensorflow as tf
import torch
import pickle
from torch import tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--pooling", type=bool, default=False)
    parser.add_argument("--min_freq", type=float, default=0.5e9)
    parser.add_argument("--max_freq", type=float, default=10e9)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--in_time", type=bool, default=False)
    args = parser.parse_args()
    return args


def filter_classes(inputs, labels, keep_classes):
    mask = np.isin(labels, keep_classes)
    filtered_inputs = inputs[mask]
    filtered_labels = labels[mask]
    return filtered_inputs, filtered_labels


def remap_labels(labels, mapping):
    mapped_labels = np.vectorize(mapping.get)(labels)
    return mapped_labels


def load_and_preprocess_data(args: argparse.Namespace):
    """This is where we load in and preprocess our data! We load in the data
        for you but you'll need to flatten the images, normalize the values and
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!"""

    # Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    keep_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    test_inputs, test_labels = filter_classes(test_inputs, test_labels, keep_classes)
    train_inputs, train_labels = filter_classes(
        train_inputs, train_labels, keep_classes
    )

    if args.pooling:
        train_inputs = pool(train_inputs)
    if args.pooling:
        test_inputs = pool(test_inputs)

    if args.in_time:
        train_inputs = train_inputs
        train_inputs = train_inputs / 255
        dig_train_inputs = []
        for i in range(train_inputs.shape[1]):
            dig_train_inputs.append(
                fm(train_inputs[: args.size, i, :], Fi=args.min_freq, Ff=args.max_freq)
            )
        dig_train_inputs = torch.tensor(np.array(dig_train_inputs))
        dig_train_inputs = dig_train_inputs.transpose(0, 1)
        dig_train_inputs = dig_train_inputs.repeat_interleave(3, dim=1)
        refined_inputs = add_zeros(dig_train_inputs, 500)
    else:
        dig_train_inputs = (
            train_inputs.reshape(-1, train_inputs.shape[-1] * train_inputs.shape[-2])
            / 255
        )[0 : args.size]
        refined_inputs = wave_transform(dig_train_inputs, args.min_freq, args.max_freq)
        refined_inputs = add_zeros(refined_inputs, 500)
    if args.in_time:
        test_inputs = test_inputs / 255
        testing_size = int(0.2 * args.size) if int(0.2 * args.size) >= 320 else 320
        test_inputs = test_inputs[:testing_size]
        dig_test_inputs = []
        for i in range(test_inputs.shape[1]):
            dig_test_inputs.append(
                fm(test_inputs[:testing_size, i, :], Fi=args.min_freq, Ff=args.max_freq)
            )
        dig_test_inputs = torch.tensor(np.array(dig_test_inputs))
        dig_test_inputs = dig_test_inputs.transpose(0, 1)
        dig_test_inputs = dig_test_inputs.repeat_interleave(3, dim=1)
        new_test_inputs = add_zeros(dig_test_inputs, 500)
    else:
        dig_test_inputs = (
            test_inputs.reshape(-1, test_inputs.shape[-1] * test_inputs.shape[-2]) / 255
        )
        new_test_inputs = wave_transform(dig_test_inputs, args.min_freq, args.max_freq)
        new_test_inputs = add_zeros(new_test_inputs, 500)

    refined_ouputs = remap_labels(train_labels[0 : args.size], label_mapping)
    testing_size = int(0.2 * args.size) if int(0.2 * args.size) >= 320 else 320
    dig_test_inputs = dig_test_inputs[0:testing_size]
    print(
        f"Refined inputs shape: {refined_inputs.shape}"
    )  # should be batch_size x width x tiemesteps x 1
    new_test_labels = remap_labels(test_labels[0:testing_size], label_mapping)
    train_labels = tensor(refined_ouputs, dtype=torch.long)
    test_labels = tensor(new_test_labels, dtype=torch.long)

    print(f"test labels shape {train_labels.shape}")
    print(f"train inputs shape {refined_inputs.shape}")
    print(f"test_inputs shape {new_test_inputs.shape}")
    print(f"test labels ahape {test_labels.shape}")

    with open(f"C:/spins/data/data.p", "wb") as pickle_file:
        pickle.dump(
            dict(
                train_inputs=refined_inputs,
                train_labels=train_labels,
                test_inputs=new_test_inputs,
                test_labels=test_labels,
                dig_train_inputs=dig_train_inputs,
                dig_test_inputs=dig_test_inputs,
            ),
            pickle_file,
        )
    print(f'Data has been dumped into {"C:/spins/data"}/data.p!')


def fm(inputs: np.array, Fi: float, Ff: float) -> np.array:
    """
    Frequency modulate the input images.

    Parameters:
        inputs (np.array): Array of shape (number of inputs, 784), where each row represents an image.
        Fi (float): Minimum frequency in Hz.
        Ff (float): Final frequency in Hz.

    Returns:
        np.array: Frequency modulated waveforms for each input image.
    """
    points_per_input = 600 // (inputs.shape[1])
    dt = 20e-12  # timestep (s)
    timesteps = inputs.shape[1] * points_per_input
    t = np.arange(0, timesteps * dt, dt)  # time vector
    modulated_wave = np.zeros((inputs.shape[0], timesteps), dtype="float32")
    pbar = tqdm(inputs)
    for i in range(inputs.shape[0]):
        pbar.set_description(f"[({i+1}/{len(inputs)})] Processing images into waves")
        pos_deriv = True
        prev = 0
        for j, pixel_intensity in enumerate(inputs[i]):
            # Calculate the corresponding frequency for this pixel
            frequency = Fi + pixel_intensity * (Ff - Fi)
            if j > 0:
                phase = np.arcsin(prev)
                if not pos_deriv:
                    phase = np.pi - phase
                modulated_wave[i, points_per_input * j : points_per_input * (j + 1)] = (
                    np.sin(2 * np.pi * frequency * t[1 : points_per_input + 1] + phase)
                )
                if np.cos(2 * np.pi * frequency * t[points_per_input] + phase) > 0:
                    pos_deriv = True
                else:
                    pos_deriv = False
                prev = np.sin(2 * np.pi * frequency * t[points_per_input] + phase)
            else:
                modulated_wave[i, points_per_input * j : points_per_input * (j + 1)] = (
                    np.sin(2 * np.pi * frequency * t[0:points_per_input])
                )
                if np.cos(2 * np.pi * frequency * t[points_per_input - 1]) > 0:
                    pos_deriv = True
                else:
                    pos_deriv = False
                prev = np.sin(2 * np.pi * frequency * t[points_per_input - 1])
    return modulated_wave


def pool(inputs: np.array):
    """Performs average pooling on image, effectively cutting down the resolution. For
    mnist this means going from 28 by 28 to 14 by 14 for 2 by 2 pooling with a stride of 2.
    """
    inputs = tensor(inputs, dtype=torch.float32)
    inputs = inputs.unsqueeze(1)
    # the output width and heigh is governed by the following equation assuming no padding:
    # w_f = (w_i + filter_width)/stride and the same for heigh
    pooling_layer = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=3, padding=0)
    pooled = pooling_layer(inputs).squeeze()
    to_return = pooled.numpy()

    return to_return


def wave_transform(inputs: np.array, min_freq, max_freq):
    t = np.arange(0, 600 * 20e-12, 20e-12)
    inputs = np.expand_dims(inputs, axis=-1)
    wave = np.sin(2 * np.pi * t * (inputs * (max_freq - min_freq) + min_freq))
    return torch.tensor(wave, dtype=torch.float32)


def add_zeros(wave: tensor, number_of_zeros):
    added_zeros = torch.cat(
        (wave, torch.zeros((wave.shape[0], wave.shape[1], number_of_zeros))), dim=-1
    )
    return added_zeros.unsqueeze(-1)


def show_image(image: np.array):
    """
    Used to show mnist image"""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    load_and_preprocess_data(parseArgs())
