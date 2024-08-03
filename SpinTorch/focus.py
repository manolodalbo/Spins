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
    parser.add_argument("--learning_rate", type=float, default=0.0001)
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
    batch_size = args.batch_size

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = num_probes  # number of probes
    for p in range(Np):
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
    """Directories"""
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    cfilm = create_solver(args, num_probes=2)
    model = cfilm

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    min_freq = 1e9
    max_freq = 10e9
    timesteps = 600
    dt = 20e-12
    zero_to_one = torch.linspace(0,1,16)
    freqs = (min_freq) + zero_to_one*(max_freq-min_freq)
    print(freqs.shape)
    t = torch.arange(0, timesteps*dt, dt).unsqueeze(0).unsqueeze(2)
    print(t.shape)
    INPUTS = torch.sin(2*torch.pi * t * freqs).transpose(0,2).to(dev)
    OUTPUTS = zero_to_one.to(dev)
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
    def mse(outputs,target):
        loss = (target - outputs)**2 / outputs.shape[0]
        return loss

    """Train the network"""
    print(INPUTS.shape)
    tic()
    model.retain_history = False
    for epoch in range(epoch_init + 1, epochs):
        optimizer.zero_grad()
        u = model(INPUTS).sum(dim=-1)
        print(f"output shape: {u.shape}")
        exit()
        loss = mse(u, OUTPUTS)
        stat_cuda("after forward")
        loss.backward()
        optimizer.step()
        stat_cuda("after backward")
        loss_iter.append(loss.item())  # store loss values
        spintorch.plot.plot_loss(loss_iter, plotdir, args.plot_name)
        plt.figure()
        plt.plot()
        print(f"Epoch finished: {epoch}")


if __name__ == "__main__":
    focus(parseArgs())
