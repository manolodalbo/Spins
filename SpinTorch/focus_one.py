"""Optimize a focusing model"""

import torch
import os
import spintorch
from spintorch.utils import tic, toc, stat_cuda
import pickle
from tqdm import tqdm
import argparse
from spintorch.multi_modal import MModel


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--Bt", type=float, default=1e-3)
    parser.add_argument("--number", type=int, default=0)
    args = parser.parse_args()
    return args


def focus(args):
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
    # geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = 2  # number of probes
    for p in range(Np):
        probes.append(
            spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
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
    model = film
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    with open(f"C:\spins\data\data.p", "rb") as data_file:
        data_dict = pickle.load(data_file)
    INPUTS = torch.tensor(data_dict["train_inputs"] * Bt).unsqueeze(-1).to(dev)
    OUTPUTS = data_dict["train_labels"]  # all classes in outputs
    print(OUTPUTS)
    OUTPUTS = OUTPUTS.to(dev)
    TEST_INPUTS = torch.tensor(data_dict["test_inputs"] * Bt).unsqueeze(-1).to(dev)
    TEST_OUTPUTS = data_dict["test_labels"].to(dev)  # desired output
    """Define optimizer and lossfunction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_init = -1
    loss_iter = []
    """Train the network"""
    print(INPUTS.shape)
    tic()
    model.retain_history = False
    high_accuracy = 0
    max_loss = 1000

    def bce(output, target_index):
        target_index = target_index.long()
        ohe = torch.nn.functional.one_hot(target_index, 2).float()
        preds = output / (output.sum(dim=-1).unsqueeze(-1))
        loss = torch.nn.functional.binary_cross_entropy(preds, ohe)
        return loss

    for epoch in range(epoch_init + 1, epochs):
        with tqdm(
            total=INPUTS.shape[0] // batch_size, desc=f"Epoch {epoch + 1}/{epochs}"
        ) as pbar:
            indices = torch.randperm(INPUTS.shape[0], device=dev)
            INPUTS = INPUTS[indices]
            OUTPUTS = OUTPUTS[indices]
            epoch_loss = 0
            epoch_accuracy = 0
            for b, b1 in enumerate(range(batch_size, INPUTS.shape[0] + 1, batch_size)):
                b0 = b1 - batch_size
                u = model(INPUTS[b0:b1])
                loss = bce(u, OUTPUTS[b0:b1])
                epoch_loss += loss.item()
                accuracy = (u.argmax(dim=-1) == OUTPUTS[b0:b1]).float().mean()
                epoch_accuracy += accuracy
                if accuracy > high_accuracy:
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss_iter": loss_iter,
                            "model_state_dict": model.state_dict(),
                        },
                        savedir + "model_highest_accuracy" + args.plot_name + ".pt",
                    )
                    high_accuracy = accuracy
                if loss.item() < max_loss:
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss_iter": loss_iter,
                            "model_state_dict": model.state_dict(),
                        },
                        savedir + "model_lowest_loss" + args.plot_name + ".pt",
                    )
                    max_loss = loss.item()
                stat_cuda("after forward")
                loss.backward()
                optimizer.step()
                stat_cuda("after backward")
                loss_iter.append(loss.item())  # store loss values
                pbar.set_description(
                    f"Batch {b + 1}/{INPUTS.shape[0]//batch_size}, Loss: {loss.item():.6f}, Accuracy: {accuracy.item():.6f}"
                )
                pbar.update(1)
                try:
                    spintorch.plot.plot_loss(loss_iter, plotdir, args.plot_name)
                except:
                    print("Plotting loss failed")
            pbar.set_postfix_str(
                f"Epoch Loss: {epoch_loss:.6f}, Epoch Accuracy: {epoch_accuracy / (b + 1):.6f}"
            )
            print(
                "Epoch finished: %d -- Loss: %.6f -- Accuracy: %f"
                % (epoch, epoch_loss, epoch_accuracy / (b + 1))
            )
        try:
            with torch.no_grad():
                total_positives = 0
                total_positive_accurate = 0
                total_test_accuracy = 0.0

                num_batches = TEST_INPUTS.shape[0] // batch_size
                for i in range(num_batches):
                    test_batch_inputs = TEST_INPUTS[
                        i * batch_size : (i + 1) * batch_size
                    ]
                    test_batch_outputs = TEST_OUTPUTS[
                        i * batch_size : (i + 1) * batch_size
                    ]

                    test_outputs = model(test_batch_inputs)
                    batch_accuracy = (
                        (test_outputs.argmax(dim=-1) == test_batch_outputs)
                        .float()
                        .mean()
                        .item()
                    )
                    total_test_accuracy += batch_accuracy

                    # Count positives and accurate positives
                    for j in range(test_outputs.shape[0]):
                        if test_batch_outputs[j] == 1:
                            total_positives += 1
                            if test_outputs[j].argmax() == 1:
                                total_positive_accurate += 1

                # Calculate average test accuracy
                avg_test_accuracy = total_test_accuracy / num_batches
                print(f"Test Accuracy: {avg_test_accuracy:.6f}")

                # Calculate positive accuracy
                if total_positives > 0:
                    positive_accuracy = total_positive_accurate / total_positives
                    print(f"Positive Accuracy: {positive_accuracy:.6f}")
                else:
                    print("No positive samples in test set.")
        except Exception as e:
            print(f"Test failed: {e}")
            toc()

            """Save model checkpoint"""
        torch.save(
            {
                "epoch": epoch,
                "loss_iter": loss_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            savedir + "model_e%d" % (epoch) + args.plot_name + ".pt",
        )


if __name__ == "__main__":
    focus(parseArgs())
