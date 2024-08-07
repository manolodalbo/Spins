"""Optimize a focusing model"""

import torch
import os
import spintorch
from spintorch.utils import tic, toc, stat_cuda
import pickle
from tqdm import tqdm
import argparse
from spintorch.multi_modal import MModel
import matplotlib.pyplot as plt


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--Bt", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def create_solver(args, num_probes):
    """Parameters"""
    dx = 50e-9  # discretization (m)
    dy = 50e-9  # discretization (m)
    dz = 20e-9  # discretization (m)
    nx = 100  # size x    (cells)
    ny = 100  # size y    (cells)

    Ms = 140e3  # saturation magnetization (A/m)
    B0 = 60e-3  # bias field (T)

    dt = 20e-12  # timestep (s)
    # batch_size = args.batch_size
    batch_size = 1

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = num_probes  # number of probes
    for p in range(Np):
        probes.append(spintorch.WaveIntensityProbe(nx - 15, (ny - Np) // 2 + p))
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    return film


def focus(args):
    Bt = args.Bt  # excitation field amplitude (T)
    learning_rate = args.learning_rate
    epochs = 50
    batch_size = args.batch_size
    """Directories"""
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    cfilm = create_solver(args, num_probes=50)
    model = cfilm

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU

    n_timesteps = 60
    # Generate 50 equally spaced values from 0 to 2*pi
    x = torch.linspace(0, 2 * torch.pi, n_timesteps)
    # Compute the sine of these values
    y = torch.sin(x) / 2 + 1 / 2
    # Print the values
    print(y.shape)
    min_freq = 0.5e9
    max_freq = 10e9

    outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
    print(outputs.shape)
    dt = 20e-12
    t = torch.arange(0, 600 * dt, dt).unsqueeze(0).unsqueeze(-1)  # time vector
    waves = torch.sin(2 * torch.pi * t * outputs).transpose(0, 2).unsqueeze(0)
    """Define optimizer and lossfunction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.linspace(0, 2 * torch.pi, 50)
    OUTPUTS = torch.cos(x)
    print("output shape")
    print(OUTPUTS.shape)
    OUTPUTS = (OUTPUTS).to(dev)
    print("output sum")
    print(OUTPUTS.abs().sum())
    print(OUTPUTS[0])
    plt.figure(figsize=(10, 6))
    plt.plot(OUTPUTS.cpu().detach().numpy(), label="output")
    plt.legend()
    plt.savefig("C:/original_spintorch/SpinTorch/plots/output.png")
    plt.close()
    INPUTS = torch.cat(
        ((waves * Bt), torch.zeros(1, waves.shape[1], 500, 1)), dim=2
    ).to(dev)
    print(f"inputs shape: {INPUTS.shape}")
    """Define optimizer and lossfunction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_init = -1
    loss_iter = []

    def bce(output, target_index):
        target_index = target_index.long()
        ohe = torch.nn.functional.one_hot(target_index, 2).float()
        preds = output / (output.sum(dim=-1).unsqueeze(-1))
        loss = torch.nn.functional.binary_cross_entropy(preds, ohe)
        return loss

    def cross_entropy(outputs, target_index):
        # target_index = target_index.long()
        # ohe = torch.nn.functional.one_hot(target_index, 2).float()
        preds = outputs / (outputs.sum(dim=-1).unsqueeze(-1))
        loss = torch.nn.functional.cross_entropy(preds, target_index)
        return loss

    def mse(outputs, target):
        sub = target - outputs
        exp = sub**2
        sum = exp.sum(dim=0)
        loss = sum / outputs.shape[0]
        return loss

    """Train the network"""
    print(INPUTS.shape)
    tic()
    model.retain_history = False
    for epoch in range(epoch_init + 1, epochs):
        optimizer.zero_grad()
        u = model(INPUTS)
        u = u.sum(dim=-1)
        u = u.squeeze()
        u = (u - u.mean()) / (u.std() * torch.sqrt(torch.tensor(2)))
        print(f"output shape: {u.shape}")
        plt.figure()
        plt.plot(u.cpu().detach().numpy())
        plt.savefig(plotdir + f"output_{epoch}.png")
        plt.close()
        loss = torch.nn.functional.mse_loss(u, OUTPUTS)
        print(loss)
        stat_cuda("after forward")
        loss.backward()
        optimizer.step()
        stat_cuda("after backward")
        loss_iter.append(loss.item())  # store loss values
        spintorch.plot.plot_loss(loss_iter, plotdir, "sinusoidal")
        print(f"Epoch finished: {epoch}")


if __name__ == "__main__":
    focus(parseArgs())
