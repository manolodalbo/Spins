import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch
from preprocess import fm


def generate_freqeuency_modulated():
    n_timesteps = 10000
    # Generate 50 equally spaced values from 0 to 2*pi
    x = torch.linspace(0, 2 * torch.pi, n_timesteps)

    # Compute the sine of these values
    y = torch.sin(x)

    # Print the values
    min_freq = 3e7
    max_freq = 30e7

    outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
    print(outputs.shape)
    dt = 20e-12
    t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
    inside = 2 * torch.pi * outputs.unsqueeze(-1) * t
    tensor = torch.sin(inside)
    print(tensor.shape)
    x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 80, tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
    plt.tick_params(axis="both", which="major", labelsize=12)

    ax.set_xlabel("t(dt=20e-12s)", fontsize=12, weight="bold")
    ax.set_ylabel("width of film(y)", fontsize=12, weight="bold")
    ax.set_zlabel("Intensity(mT)", fontsize=12, weight="bold")
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio is in the form [x, y, z]

    plt.show()


def generate_common():
    # Generate 50 equally spaced values from 0 to 2*pi
    freq = 0.8e9
    dt = 20e-12
    t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
    tensor = torch.sin(2 * torch.pi * t * freq)
    print(tensor.shape)
    tensor = tensor.repeat(80, 1)
    x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 80, tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
    ax.set_xlabel("t(dt = 20e-12s)", fontsize=12)
    ax.set_ylabel("width of film(y)", fontsize=12)
    ax.set_zlabel("Intensity (mT)", fontsize=12)
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio is in the form [x, y, z]
    plt.show()


def simple():
    x = np.linspace(0, 1, 100)

    # Define the function f(x) = x
    y = x

    # Create the plot
    plt.figure()
    plt.plot(x, y, label="f(x) = x")

    # Add labels and title with increased font size
    plt.xlabel("x", fontsize=14)
    plt.ylabel("f(x)", fontsize=14)

    # Add legend with increased font size
    plt.legend(fontsize=12)

    # Increase tick label size
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Show the plot
    plt.grid(True)
    plt.show()


def simple_scaled():
    x = np.linspace(1e9, 1e10, 6)

    # Define the function y = x and scale y between 0 and 1
    y = (x - 1e9) / (1e10 - 1e9)
    # Create the plot
    plt.figure()
    plt.plot(x, y, "o")

    # Add labels and title with increased font size
    plt.xlabel("freq(Hz)", fontsize=14, weight="bold")
    plt.ylabel("f(x)", fontsize=14, weight="bold")

    # Add legend with increased font size

    # Increase tick label size
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Show the plot
    plt.grid(True)
    plt.show()


def simple_sin():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x", fontsize=14, weight="bold")
    plt.ylabel("sin(x)", fontsize=14, weight="bold")
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(True)
    plt.show()


def sin_scaled():
    x = np.linspace(0, 2 * np.pi, 100)
    max_freq = 10e9
    min_freq = 0.5e9
    y = ((np.sin(x) + 1) / 2) * (max_freq - min_freq) + min_freq
    plt.figure()
    x = np.linspace(0, 80, 100)
    plt.plot(x, y)
    plt.xlabel("width(y)", fontsize=14, weight="bold")
    plt.ylabel("freq(Hz)", fontsize=14, weight="bold")
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(True)
    plt.show()


def reandom_graph():
    randoms = np.random.rand(60)
    plt.figure()
    plt.plot(randoms, "o")
    plt.xlabel("probes", fontsize=12)
    plt.ylabel("intensities", fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(True)
    plt.show()


def simple_cos():
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.array(
        ((torch.cos(torch.tensor(x)) + 1) / 2) * 10e10
        + torch.normal(mean=0, std=0.5e10, size=(10,))
    )
    y = (y - y.mean()) / (np.sqrt(2) * y.std())
    x = np.arange(0, 10)
    plt.figure()
    plt.plot(x, y, "o")
    plt.title("Normalized Output")
    plt.xlabel("probes", fontsize=14, weight="bold")
    plt.ylabel("intensity", fontsize=14, weight="bold")
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(True)
    plt.show()


def frequency_over_space():
    x = np.array([3 / 5, 0, 0, 3 / 5, 1 / 5, 5 / 5])
    max_freq = 5e9
    min_freq = 0.5e9
    x = min_freq + x * (max_freq - min_freq)
    plt.figure()
    plt.plot(x, "o-")
    plt.xlabel("width(y)", fontsize=12)
    plt.ylabel("frequency(Hz)", fontsize=12)
    plt.show()
    plt.close()


def frequency_modulated2d():
    x = np.linspace(0, 2 * np.pi, 600)
    max_freq = 5e9
    min_freq = 0.5e9
    y = np.array([0, 1 / 2, 2])
    frequency_modulated2d = fm(np.expand_dims(y, axis=0), min_freq, max_freq, 200)
    plt.figure()
    plt.plot(frequency_modulated2d[0])
    plt.xlabel("t (dt = 20e-12)", fontsize=14, weight="bold")
    plt.ylabel("Intensity (mT)", fontsize=14, weight="bold")
    plt.show()
    plt.close()


def generate_fm_3d():
    # Generate 50 equally spaced values from 0 to 2*pi
    x = np.linspace(0, 2 * np.pi, 600)
    max_freq = 5e9
    min_freq = 0.5e9
    y = (np.sin(x) + 1) / 2
    frequency_modulated2d = fm(np.expand_dims(y, axis=0), min_freq, max_freq, 1)
    print(frequency_modulated2d.shape)
    tensor = np.tile(frequency_modulated2d, (80, 1))
    print(tensor.shape)
    x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 80, tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
    ax.set_xlabel("t(dt = 20e-12s)", fontsize=12)
    ax.set_ylabel("width of film(y)", fontsize=12)
    ax.set_zlabel("Intensity (mT)", fontsize=12)
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio is in the form [x, y, z]
    plt.show()


def generate_fm_3d_real():
    # Generate 50 equally spaced values from 0 to 2*pi
    y = np.array(
        [
            [1 / 9, 3 / 9, 8 / 9],
            [9 / 9, 0, 2 / 9],
            [1 / 9, 0, 0],
            [0, 3 / 9, 7 / 9],
            [3 / 9, 1 / 9, 2 / 9],
            [5 / 9, 5 / 9, 4 / 9],
        ]
    )
    max_freq = 5e9
    min_freq = 0.5e9
    tensor = fm(y, min_freq, max_freq, 200)
    print(tensor.shape)
    x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 6, tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
    ax.set_xlabel("t(dt = 20e-12s)", fontsize=12)
    ax.set_ylabel("width of film(y)", fontsize=12)
    ax.set_zlabel("Intensity (mT)", fontsize=12)
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio is in the form [x, y, z]
    plt.show()


def generate_fm_3d_nice():
    y = torch.tensor([0 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 1 / 2])

    # Print the values
    min_freq = 3e7
    max_freq = 30e7

    outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
    dt = 20e-12
    t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
    inside = 2 * torch.pi * outputs.unsqueeze(-1) * t
    tensor = torch.sin(inside)
    print(tensor.shape)
    x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 80, tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
    plt.tick_params(axis="both", which="major", labelsize=12)

    ax.set_xlabel("t(dt=20e-12s)", fontsize=12, weight="bold")
    ax.set_ylabel("width of film(y)", fontsize=12, weight="bold")
    ax.set_zlabel("Intensity(mT)", fontsize=12, weight="bold")
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio is in the form [x, y, z]

    plt.show()


simple_scaled()
