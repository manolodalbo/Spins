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
import numpy as np
from scipy.fft import fft, fftfreq
from spintorch.plot import (
    wave_integrated,
    wave_snapshot,
    wave_video,
    wave_animation,
    wave_intensity_animation,
    save_wave_intensity,
    save_wave_intensity_parallel,
)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--Bt", type=float, default=1e-3)
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
    batch_size = 1

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = 1  # number of probes
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
    model = create_solver(args)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    dt = 20e-12
    timesteps = 300
    t = (
        torch.arange(0, timesteps * dt, dt, device=dev).unsqueeze(0).unsqueeze(2)
    )  # time vector
    t_longer = torch.arange(0, 700 * dt, dt, device=dev).unsqueeze(0).unsqueeze(2)
    # INPUTS = Bt * torch.cat((torch.ones(1, 100, 1), torch.zeros(1, 1000, 1)), dim=1).to(
    #     dev
    # )  # excitation field
    INPUTS = torch.cat(
        (
            Bt * torch.sin(2 * torch.pi * 3e9 * t),
            torch.zeros((1, 1000, 1), device=dev),
            # Bt * torch.sin(2 * torch.pi * 1e9 * t),
            # torch.zeros((1, 1000, 1), device=dev),
        ),
        dim=1,
    ).to(dev)
    # INPUTS2 = Bt * torch.sin(2 * torch.pi * 3e9 * t_longer).to(dev)  # excitation field
    x = torch.linspace(0, 2 * torch.pi, 10).to(dev)
    OUTPUTS = torch.cos(x).to(dev)
    print(f"outputs mean: {OUTPUTS.mean()} outputs std: {OUTPUTS.std()}")
    print(INPUTS.shape)
    tic()
    model.retain_history = True
    epochs = 40
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_iter = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(INPUTS)[:, :, 600:1100]
        t_per_bucket = 50
        buckets = outputs.view(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] // t_per_bucket,
            t_per_bucket,
        ).sum(dim=-1)

        norm_buckets = (
            (buckets - buckets.mean())
            * torch.tensor(0.7746, device="cuda")
            / (buckets.std())
        ) + 0.1
        plt.figure(figsize=(10, 6))
        plt.plot(norm_buckets[0, 0, :].detach().cpu().numpy())
        plt.title("Bucket Outputs")
        plt.savefig(f"bucket_output_{epoch}.png")
        plt.close()
        loss = torch.nn.functional.mse_loss(norm_buckets.squeeze(), OUTPUTS)
        loss_iter.append(loss.item())
        plt.figure(figsize=(10, 6))
        plt.plot(loss_iter, "o-")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("loss_damping.png")
        plt.close()
        loss.backward()
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
                model, plotdir=plotdir + "geometry.png", epoch=epoch
            )
            spintorch.plot.damping(model, plotdir=plotdir + "damping.png")
        print(f"Epoch {epoch} Loss: {loss.item()}")
    plt.figure(figsize=(10, 6))
    plt.plot(outputs[0, 0, :].detach().cpu().numpy())
    plt.title("Output")
    plt.savefig("init_out.png")
    plt.close()
    if model.retain_history:
        with torch.no_grad():
            mz = (
                torch.stack(model.m_history, 1)[
                    0,
                    :,
                    2,
                ]
                - model.m0[
                    0,
                    2,
                ]
                .unsqueeze(0)
                .cpu()
            )
            # wave_integrated(model, mz, plotdir + "wave_integrated_7200.png")
            # save_wave_intensity(model, mz, "plots/to_view/")
            # wave_intensity_animation(model, mz, "plots/video")
            save_wave_intensity_parallel(model, mz, "plots/spike/")


if __name__ == "__main__":
    focus(parseArgs())
