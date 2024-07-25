from functools import reduce
import numpy as np
import torch
import matplotlib.pyplot as plt


def turn_into_wave(inputs, embedding_matrix):
    inputs = embedding_matrix[
        inputs.long()
    ]  # should return batch_size x words x embed_size
    print(f"input: {inputs[0]}")
    inputs = (inputs - 7.8e6) / 2.26e6
    inputs = inputs.transpose(-1, -2)
    to_return = fm_simpler(inputs)
    to_return = torch.flatten(to_return, start_dim=2, end_dim=-1).unsqueeze(-1)
    return to_return


def fm_simpler(inputs: torch.tensor, Fi: float = 0.5e9, Ff: float = 10e9):
    points_per_input = 600 / inputs.shape[-1]
    dt = 20e-12
    t = (
        torch.arange(0, points_per_input * dt, dt, device=inputs.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    inputs = torch.sigmoid(inputs)  # scaled between 0 and 1
    middle = t * ((Fi) + inputs.unsqueeze(-1) * (Ff - Fi))
    to_return = torch.sin(2 * torch.pi * middle)
    return to_return


def fm(inputs: torch.tensor, Fi: float = 0.5e9, Ff: float = 10e9) -> torch.tensor:
    """
    Frequency modulate the input images.

    Parameters:
        inputs (np.array): Array of shape (number of inputs, 784), where each row represents an image.
        Fi (float): Minimum frequency in Hz.
        Ff (float): Final frequency in Hz.

    Returns:
        np.array: Frequency modulated waveforms for each input image.
    """
    inputs = torch.sigmoid(0.3 * inputs)
    points_per_input = 600 // inputs.shape(1)
    dt = 20e-12  # timestep (s)
    timesteps = 600
    t = torch.arange(0, timesteps * dt, dt)  # time vector
    modulated_wave = torch.zeros((inputs.shape[0], timesteps), dtype=torch.float32)
    for i in range(inputs.shape(0)):
        pos_deriv = True
        prev = 0
        for j, pixel_intensity in enumerate(inputs[i]):
            # Calculate the corresponding frequency for this pixel
            frequency = Fi + pixel_intensity * (Ff - Fi)
            if j > 0:
                phase = torch.arcsin(prev)
                if not pos_deriv:
                    phase = torch.pi - phase
                modulated_wave[i, points_per_input * j : points_per_input * (j + 1)] = (
                    torch.sin(
                        2 * torch.pi * frequency * t[1 : points_per_input + 1] + phase
                    )
                )
                if (
                    torch.cos(2 * torch.pi * frequency * t[points_per_input] + phase)
                    > 0
                ):
                    pos_deriv = True
                else:
                    pos_deriv = False
                prev = torch.sin(2 * torch.pi * frequency * t[points_per_input] + phase)
            else:
                modulated_wave[i, points_per_input * j : points_per_input * (j + 1)] = (
                    np.sin(2 * np.pi * frequency * t[0:points_per_input])
                )
                if torch.cos(2 * torch.pi * frequency * t[points_per_input - 1]) > 0:
                    pos_deriv = True
                else:
                    pos_deriv = False
                prev = torch.sin(2 * torch.pi * frequency * t[points_per_input - 1])
    return modulated_wave


def get_data(train_file, test_file):
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    with open(train_file, "r") as txt:
        train_as_string = txt.read()
    with open(test_file, "r") as txt2:
        test_as_string = txt2.read()
    train_data = train_as_string.lower().split()
    test_data = test_as_string.lower().split()
    unique_words = sorted(set(train_data))
    vocabulary = {word: key for key, word in enumerate(unique_words)}
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    # assert all(0 <= value < vocab_size for value in vocabulary.values())

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data = list(map(lambda x: vocabulary[x], test_data))
    return train_data, test_data, vocabulary
