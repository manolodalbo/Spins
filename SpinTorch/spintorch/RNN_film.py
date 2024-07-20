import torch.nn as nn
import SpinTorch.spintorch as spintorch


class RNN_film(nn.Module):
    def __init__(self, embed_size=80, batch_size=64):
        super(RNN_film, self).__init__()
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
        Np = embed_size  # number of probes
        for p in range(Np):
            probes.append(spintorch.WaveIntensityProbe(nx - 15, int((ny - Np) / 2) + p))
        self.film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)

    def forward(self, inputs):
        return self.film(inputs)
