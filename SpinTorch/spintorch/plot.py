import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
from .geom import WaveGeometryMs, WaveGeometry
from .solver import MMSolver
import torch
import warnings
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import multiprocessing as mp
import os

warnings.filterwarnings("ignore", message=".*No contour levels were found.*")


mpl.use(
    "Agg",
)  # uncomment for plotting without GUI
mpl.rcParams["figure.figsize"] = [8.0, 6.0]
mpl.rcParams["figure.dpi"] = 600


def plot_loss(loss_iter, plotdir, unique_id):
    fig = plt.figure()
    plt.plot(loss_iter, "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir + "loss" + unique_id + ".png")
    plt.close(fig)


def plot_accuracy(acc_iter, plotdir):
    fig = plt.figure()
    plt.plot(acc_iter, "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir + "accuracy.png")
    plt.close(fig)


def plot_output(u, p, epoch, plotdir):
    fig = plt.figure()
    plt.bar(range(1, 1 + u.size()[0]), u.detach().cpu().squeeze(), color="k")
    plt.xlabel("output number")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir + "output_epoch%d_X%d.png" % (epoch, p))
    plt.close(fig)


def _plot_probes(probes, ax):
    markers = []
    for i, probe in enumerate(probes):
        x, y = probe.coordinates()
        (marker,) = ax.plot(
            x,
            y,
            ".",
            markeredgecolor="none",
            markerfacecolor="k",
            markersize=4,
            alpha=0.8,
        )
        markers.append(marker)
    return markers


def _plot_sources(sources, ax):
    markers = []
    for i, source in enumerate(sources):
        x, y = source.coordinates()
        (marker,) = ax.plot(
            x,
            y,
            ".",
            markeredgecolor="none",
            markerfacecolor="g",
            markersize=4,
            alpha=0.8,
        )
        markers.append(marker)
    return markers


def geometry(
    model, ax=None, outline=False, outline_pml=True, epoch=0, plotdir="", plotname=""
):

    geom = model.geom
    probes = model.probes
    sources = model.sources
    A = model.Alpha()[
        0,
        0,
    ].squeeze()
    alph = A.min().cpu().numpy()
    B = geom.B[1,].detach().cpu().numpy().transpose()
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    markers = []
    if not outline:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label="Saturation magnetization (A/m)")
        else:
            h1 = ax.imshow(B * 1e3, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label="Magnetic field (mT)")
    else:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            ax.contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
        else:
            ax.contour(B, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(
            b_boundary,
            levels=[alph * 1.0001],
            colors=["k"],
            linestyles=["dotted"],
            linewidths=[0.75],
            alpha=1,
        )

    markers += _plot_probes(probes, ax)
    markers += _plot_sources(sources, ax)

    if plotdir:
        fig.savefig(plotdir + "geometry_epoch%d" % (epoch) + plotname + ".png")
        plt.close(fig)


def damping(model, plotdir=""):
    markers = []
    damping = model.Alpha
    # damping_field = (
    #     (
    #         torch.sigmoid(damping.Rho.detach().cpu())
    #         * (damping.alpha_real_max - damping.alpha_min)
    #         + damping.alpha_min
    #     )
    #     .squeeze()
    #     .numpy()
    # )
    damping_field = damping.Rho.detach().cpu().squeeze().numpy().transpose()
    damping_field = damping_field[10:90, 10:90]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    h = ax.imshow(damping_field, origin="lower", cmap=plt.cm.viridis)
    plt.colorbar(h, ax=ax, label="Damping field")
    # markers += _plot_probes(model.probes, ax)
    # markers += _plot_sources(model.sources, ax)
    if plotdir:
        fig.savefig(plotdir + "damping.png")
        plt.close(fig)


def wave_integrated(model, m_history, filename=""):

    m_int = m_history.pow(2).sum(dim=0).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    vmax = m_int.max()
    print(f"vmax: {vmax}")
    h = ax.imshow(
        m_int,
        cmap=plt.cm.viridis,
        origin="lower",
        norm=LogNorm(vmin=vmax * 0.01, vmax=vmax),
    )
    plt.colorbar(h)
    geometry(model, ax=ax, outline=True)

    if filename:
        fig.savefig(filename)
        plt.close(fig)


def wave_snapshot(model, m_snap, filename="", clabel="m"):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    m_t = m_snap.cpu().numpy().transpose()
    h = axs.imshow(m_t, cmap=plt.cm.RdBu_r, origin="lower", norm=plt.LogNorm())
    geometry(model, ax=axs, outline=True)
    plt.colorbar(h, ax=axs, label=clabel, shrink=0.80)
    axs.axis("image")
    if filename:
        fig.savefig(filename)
        plt.close(fig)


def wave_video(model, m_snapshots, plotdir):
    for t in range(m_snapshots.shape[0]):
        wave_snapshot(model, m_snapshots[t], plotdir + "wave_t%03d.png" % t)
    plt.show()


def wave_animation(model, m_snapshots, plotdir):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    artists = []
    print(f"snapshots shape: {m_snapshots.shape}")
    vmin = -0.01
    vmax = 0.08
    print(vmin)
    print(vmax)
    print(m_snapshots)
    for i in tqdm(range(m_snapshots.shape[0])):
        m_t = m_snapshots[i].cpu().numpy().transpose()
        h = axs.imshow(
            m_t,
            cmap=plt.cm.RdBu_r,
            origin="lower",
            norm=LogNorm(vmin, vmax),
        )
        geometry(model, ax=axs, outline=True)
        artists.append([h])
    ani = animation.ArtistAnimation(fig, artists, interval=50, blit=True)
    ani.save(plotdir + "wave_animation.gif", writer=PillowWriter(fps=20))


def wave_intensity_animation(model, m_snapshots, plotdir):
    m_int = m_snapshots.pow(2).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    vmax = m_int.max()
    h = ax.imshow(
        m_snapshots[0]
        .pow(2)
        .numpy()
        .transpose(),  # Use the first frame for initialization
        cmap=plt.cm.viridis,
        origin="lower",
        norm=LogNorm(vmin=vmax * 0.01, vmax=vmax),
    )
    plt.colorbar(h)  # Add the colorbar once outside the update function

    def update(frame):
        h.set_array(m_snapshots[600 + frame].pow(2).numpy().transpose())
        # geometry(model, ax=ax, outline=True)

    writer = animation.FFMpegWriter(
        fps=20, codec="libx264", bitrate=1800, extra_args=["-preset", "fast"]
    )

    print("animating...")
    ani = animation.FuncAnimation(fig, update, frames=60, interval=100)
    print("saving...")
    ani.save("C://spins//Spins//animation_new_fast.mp4", writer=writer)


def save_wave_intensity(model, m_snapshots, plotdir):
    m_int = m_snapshots.pow(2).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 4))
    vmax = m_int.max()
    geometry(model, ax=ax, outline=True)
    for i in tqdm(range(m_snapshots.shape[0])):
        h = ax.imshow(
            m_snapshots[i]
            .pow(2)
            .numpy()
            .transpose(),  # Use the first frame for initialization
            cmap=plt.cm.viridis,
            origin="lower",
            norm=LogNorm(vmin=vmax * 0.01, vmax=vmax),
        )
        fig.savefig(plotdir + "wave_intensity_%03d.png" % i)
        plt.close(fig)


def save_frame(args):
    i, snapshot, plotdir, vmax = args
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 4))
    # geometry(model, ax=ax, outline=True)
    ax.text(
        0.05,
        0.95,
        f"timestep {i:03d}",
        color="white",
        fontsize=12,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    h = ax.imshow(
        snapshot.pow(2).numpy().transpose(),
        cmap=plt.cm.viridis,
        origin="lower",
        norm=LogNorm(vmin=vmax * 0.01, vmax=vmax),
    )
    fig.savefig(os.path.join(plotdir, f"wave_intensity_{i:03d}.png"))
    plt.close(fig)


def save_wave_intensity_parallel(model, m_snapshots, plotdir):
    m_int = m_snapshots.pow(2).detach().cpu().numpy().transpose()
    vmax = m_int.max()

    args = [(i, m_snapshots[i], plotdir, vmax) for i in range(m_snapshots.shape[0])]

    with mp.Pool() as pool:
        list(tqdm(pool.imap(save_frame, args), total=len(args)))
