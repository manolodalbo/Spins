"""Optimize a focusing model"""

import torch
import os
import spintorch
from spintorch.utils import tic, toc, stat_cuda
import pickle
import argparse
from spintorch.multi_modal import MModel
from spintorch.multi_modal_integrated import IModel
from spintorch.integrating_model import OldModel
import matplotlib.pyplot as plt

# previous runs: test accuracy after epoch 0: 65%, epoch 1: 71% epoch2: 66%


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--Bt", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def create_solver(args):
    """Parameters"""
    dx = 50e-9  # discretization (m)
    dy = 50e-9  # discretization (m)
    dz = 20e-9  # discretization (m)
    nx = 100  # size x    (cells)
    ny = 100  # size y    (cells)

    Ms = 140e3  # saturation magnetization (A/m)
    B0 = 60e-3  # bias field (T)

    dt = 20e-12  # timestep (s)
    batch_size = args.batch_size

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = 10  # number of probes
    for p in range(Np):
        # probes.append(spintorch.WaveIntensityProbe(nx - 15, ((ny - Np) // 2) + p))
        probes.append(
            spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    return film


def focus(args):
    Bt = args.Bt  # excitation field amplitude (T)
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    max_freq = 10e9
    min_freq = 0.5e9
    """Directories"""
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    model = OldModel(create_solver(args), output_size=70)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    dt = 20e-12
    timesteps = 600
    t = (
        torch.arange(0, timesteps * dt, dt, device=dev).unsqueeze(0).unsqueeze(2)
    )  # time vector 1 x 1 x timesteps x 1
    with open("C:\spins\data\data.p", "rb") as data_file:
        data_dict = pickle.load(data_file)
    INPUTS = (
        (data_dict["train_inputs"]).unsqueeze(-1).unsqueeze(-1).to(dev)
    )  # 640 x 81 x 1 x 1
    INPUTS = (min_freq) + INPUTS * (max_freq - min_freq)
    OUTPUTS = data_dict["train_labels"].to(dev)  # desired output
    INPUTS = INPUTS * t
    INPUTS = (
        torch.cat(
            (
                Bt * torch.sin(2 * torch.pi * INPUTS),
                torch.zeros((INPUTS.shape[0], INPUTS.shape[1], 800, 1), device=dev),
            ),
            dim=2,
        )
    ).to(dev)
    TEST_INPUTS = (data_dict["test_inputs"]).unsqueeze(-1).unsqueeze(-1).to(dev)
    TEST_INPUTS = (min_freq) + TEST_INPUTS * (max_freq - min_freq)
    TEST_INPUTS = TEST_INPUTS * t
    TEST_INPUTS = (
        torch.cat(
            (
                Bt * torch.sin(2 * torch.pi * TEST_INPUTS),
                torch.zeros(
                    (TEST_INPUTS.shape[0], TEST_INPUTS.shape[1], 800, 1), device=dev
                ),
            ),
            dim=2,
        )
    ).to(dev)
    TEST_OUTPUTS = data_dict["test_labels"].to(dev)  # desired output

    print(INPUTS.shape)
    print(TEST_INPUTS.shape)
    print(OUTPUTS.shape)
    print(TEST_OUTPUTS.shape)
    tic()
    model.retain_history = True
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_iter = []

    def loss_fn(preds, labels):
        epsilon = 1e-8
        log_preds = torch.log(preds + epsilon)
        to_return = torch.nn.functional.nll_loss(log_preds, labels)
        return to_return

    for epoch in range(epochs):
        # Randomize inputs and outputs
        indices = torch.randperm(len(INPUTS))
        INPUTS = INPUTS[indices]
        OUTPUTS = OUTPUTS[indices]
        for i in range(0, len(INPUTS) - batch_size + 1, batch_size):
            optimizer.zero_grad()
            outputs = model(INPUTS[i : i + batch_size])
            loss = loss_fn(outputs, OUTPUTS[i : i + batch_size])
            accuracy = (
                (outputs.argmax(dim=-1) == OUTPUTS[i : i + batch_size]).float().mean()
            )
            print(
                f"epoch: {epoch}, batch: {i//batch_size} loss: {loss.item()}, accuracy: {accuracy.item()}"
            )
            loss_iter.append(loss.item())
            plt.figure(figsize=(10, 6))
            plt.plot(loss_iter, "o-")
            plt.title("Loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.savefig("loss_original.png")
            plt.close()
            loss.backward()
            torch.save(
                {
                    "epoch": epoch,
                    "loss_iter": loss_iter,
                    "model state dict": model.state_dict(),
                },
                savedir + "amp_model.pt",
            )
            # for name, param in model.named_parameters():
            #     if name == "geom.rho_param":
            #         print(param)
            #         print("parameter gradient:")
            #         print(param.grad)
            #     else:
            #         print(f"name missed: {name}")
            optimizer.step()
            with torch.no_grad():
                spintorch.plot.geometry(
                    model.film, plotdir=plotdir + "geometry.png", epoch=-1
                )
            #     spintorch.plot.damping(model, plotdir=plotdir + "damping.png")
        with torch.no_grad():
            accuracies = []
            for j in range(0, len(TEST_INPUTS) - batch_size + 1, batch_size):
                outputs = model(TEST_INPUTS[j : j + batch_size])
                accuracy = (
                    (outputs.argmax(dim=-1) == TEST_OUTPUTS[j : j + batch_size])
                    .float()
                    .mean()
                )
                accuracies.append(accuracy.item())
            print("epoch testing accuracy:")
            print(sum(accuracies) / len(accuracies))


def extract_average_frequency(signals: torch.Tensor, dt: float):
    signal_length = signals.shape[-1]

    # Compute FFT along the last dimension
    fft_result = torch.abs(torch.fft.fft(signals, dim=-1))

    # Apply the threshold
    fft_result[fft_result < 10] = 0

    # Calculate the sampling rate
    sampling_rate = 1 / dt

    # Compute frequency values for each signal
    freq = torch.fft.fftfreq(signal_length, 1 / sampling_rate, device=signals.device)

    # Ensure freq is correctly shaped to broadcast over all preceding dimensions
    freq = freq.view(*([1] * (signals.ndim - 1)), -1)

    # Compute the weighted average of frequencies
    mult = freq * fft_result
    average = mult[..., : signal_length // 2].sum(dim=-1) / fft_result[
        ..., : signal_length // 2
    ].abs().sum(dim=-1)

    return average


# Example usage:
# signals = torch.randn(batch_size, num_signals, signal_length)
# avg_freqs = extract_average_frequency(signals, dt)


if __name__ == "__main__":
    focus(parseArgs())
