"""Optimize a focusing model"""

import torch
import os
import spintorch
import optuna
from spintorch.utils import tic, toc, stat_cuda


def create_solver(batch_size, num_probes):
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
    Np = num_probes  # number of probes
    for p in range(Np):
        probes.append(
            spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    return film


def objective(trial):
    Bt = 0.01  # excitation field amplitude (T)
    # learning_rate = trial.suggest_int("lr", 0.0001, 0.1)
    learning_rate = 0.001
    epochs = 20
    """Directories"""
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    model = create_solver(2, 2)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    # timesteps_between = trial.suggest_int("timesteps between inputs", 25, 700)
    # timesteps = trial.suggest_int("timesteps", 50, 100)
    # f1 = trial.suggest_int("f1", 0.1e9, 10e9)
    # f2 = trial.suggest_int("f2", 0.1e9, 10e9)
    # f3 = trial.suggest_int("f3", 0.1e9, 10e9)
    timesteps_between = 300
    timesteps = 200
    f1 = 2e9
    f2 = 3e9
    f3 = 4e9
    dt = 20e-12
    t = torch.arange(0, timesteps * dt, dt, device=dev).unsqueeze(0).unsqueeze(2)
    FIRST_INPUT = torch.cat(
        (
            Bt * torch.sin(2 * torch.pi * f1 * t),
            torch.zeros((1, timesteps_between, 1), device=dev),
            Bt * torch.sin(2 * torch.pi * f3 * t),
            torch.zeros((1, 600, 1), device=dev),
        ),
        dim=1,
    )
    SECOND_INPUT = torch.cat(
        (
            Bt * torch.sin(2 * torch.pi * f2 * t),
            torch.zeros((1, timesteps_between, 1), device=dev),
            Bt * torch.sin(2 * torch.pi * f3 * t),
            torch.zeros((1, 600, 1), device=dev),
        ),
        dim=1,
    )
    INPUTS = torch.cat((FIRST_INPUT, SECOND_INPUT), dim=0)
    OUTPUTS = torch.tensor([0, 1]).to(dev)  # desired output
    """Define optimizer and lossfunction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_init = -1
    loss_iter = []
    """Train the network"""
    model.retain_history = False

    def my_loss(output, target_index):
        target_value = output[:, target_index]
        loss = output.sum(dim=1) / target_value - 1
        return (loss.sum() / loss.size()[0]).log10()

    for epoch in range(epoch_init + 1, epochs):
        u = model(INPUTS)[:, :, timesteps + timesteps_between + 500 :].sum(dim=-1)
        # u = u / u.sum(dim=-1).unsqueeze(-1)
        loss = my_loss(u, OUTPUTS)
        loss.backward()
        optimizer.step()
        loss_iter.append(loss.item())  # store loss values
        # spintorch.plot.plot_loss(loss_iter, plotdir=plotdir, unique_id="first")
        # print(f"epoch: {epoch} loss:{loss.item()}")
        with torch.no_grad():
            spintorch.plot.geometry(
                model, plotdir=plotdir + "geometry_recurrent.png", epoch=-1
            )
    return loss.item()


if __name__ == "__main__":
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=100)

    # print("Best parameters: ", study.best_params)
    # print("Best loss: ", study.best_value)
    objective(None)
