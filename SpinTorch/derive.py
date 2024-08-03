import spintorch
import os
import torch
def create_solver(batch_size):
    """Parameters"""
    dx = 50e-9  # discretization (m)
    dy = 50e-9  # discretization (m)
    dz = 20e-9  # discretization (m)
    nx = 100  # size x    (cells)
    ny = 100  # size y    (cells)

    Ms = 140e3  # saturation magnetization (A/m)
    B0 = 60e-3  # bias field (T)

    dt = 20e-12  # timestep (s)
    batch_size = batch_size

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = 50  # number of probes
    for p in range(Np):
        probes.append(
            spintorch.WaveIntensityProbe(nx - 15,(ny - Np)//2 + p)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    return film
def focus():
    Bt = 0.01
    learning_rate = 0.01
    batch_size = 1
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    model = create_solver(batch_size=batch_size)
    n_timesteps = 80
    # Generate 50 equally spaced values from 0 to 2*pi
    x = torch.linspace(0, 2 * torch.pi, n_timesteps)

    # Compute the sine of these values
    y = torch.sin(x)

    # Print the values
    print(y.shape)
    min_freq = 0.5e9
    max_freq = 10e9

    outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
    dt = 20e-12
    t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
    inside = 2 * torch.pi * outputs.unsqueeze(-1) * t
    inputs = torch.sin(inside).unsqueeze(-1).unsqueeze(0)
    print(inputs.shape)
focus()