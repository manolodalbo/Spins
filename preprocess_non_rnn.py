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
    keep_classes = [6, 7]
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    test_inputs, test_labels = filter_classes(test_inputs, test_labels, keep_classes)
    train_inputs, train_labels = filter_classes(
        train_inputs, train_labels, keep_classes
    )
    train_inputs = pool(train_inputs)
    print("shape of train inputs after pooling:")
    print(train_inputs.shape)
    test_inputs = pool(test_inputs)
    dig_train_inputs = (
        train_inputs.reshape(-1, train_inputs.shape[-1] * train_inputs.shape[-2]) / 255
    )[0 : args.size]
    dig_test_inputs = (
        test_inputs.reshape(-1, test_inputs.shape[-1] * test_inputs.shape[-2]) / 255
    )
    refined_inputs = wave_transform(dig_train_inputs, args.min_freq, args.max_freq)
    refined_inputs = add_zeros(refined_inputs, 500)
    refined_ouputs = remap_labels(train_labels[0 : args.size], label_mapping)
    testing_size = int(0.2 * args.size) if int(0.2 * args.size) >= 320 else 320
    dig_test_inputs = dig_test_inputs[0:testing_size]
    new_test_inputs = wave_transform(dig_test_inputs, args.min_freq, args.max_freq)
    new_test_inputs = add_zeros(new_test_inputs, 500)
    print(f"Refined inputs shape: {refined_inputs.shape}")
    new_test_labels = remap_labels(test_labels[0:testing_size], label_mapping)
    train_labels = tensor(refined_ouputs, dtype=torch.long)
    test_labels = tensor(new_test_labels, dtype=torch.long)
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
