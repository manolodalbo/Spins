"""Optimize a focusing model"""

import torch
import os
import spintorch
from spintorch.utils import tic, toc, stat_cuda
import pickle
from tqdm import tqdm
from spintorch.multi_modal import MModel
from optuna.trial import TrialState
import optuna


def create_solver(outputs):
    dx = 50e-9  # discretization (m)
    dy = 50e-9  # discretization (m)
    dz = 20e-9  # discretization (m)
    nx = 100  # size x    (cells)
    ny = 100  # size y    (cells)

    Ms = 140e3  # saturation magnetization (A/m)
    B0 = 60e-3  # bias field (T)
    dt = 20e-12  # timestep (s)

    batch_size = 64

    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = outputs  # number of probes
    for p in range(Np):
        probes.append(
            spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    return film


def objective(trial):
    Bt = trial.suggest_float("Bt", 0.005, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.5)
    epochs = 1
    batch_size = 64
    """Directories"""
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    savedir = "models/" + basedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    integrating_film = create_solver(2)
    cat_film = create_solver(2)
    model = MModel(integrating_film, cat_film)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print("Running on", dev)
    model.to(dev)  # sending model to GPU/CPU
    with open("C:\spins\data\data.p", "rb") as data_file:
        data_dict = pickle.load(data_file)
    INPUTS = torch.tensor(data_dict["train_inputs"] * Bt).unsqueeze(-1)
    INPUTS = torch.cat((INPUTS, torch.zeros(INPUTS.shape[0], 1000, 1)), dim=1).to(dev)
    print(f"inputs shape: {INPUTS.shape}")
    OUTPUTS = data_dict["train_labels"].to(dev)  # desired output
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

    def bce(output, target_index):
        target_index = target_index.long()
        ohe = torch.nn.functional.one_hot(target_index, 2).float()
        print(output)
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
                optimizer.zero_grad()
                b0 = b1 - batch_size
                u = model(INPUTS[b0:b1])
                print(f"output shape: {u.shape}")
                loss = bce(u, OUTPUTS[b0:b1])
                epoch_loss += loss.item()
                accuracy = (u.argmax(dim=-1) == OUTPUTS[b0:b1]).float().mean()
                epoch_accuracy += accuracy
                stat_cuda("after forward")
                loss.backward()
                optimizer.step()
                stat_cuda("after backward")
                loss_iter.append(loss.item())  # store loss values
                pbar.set_description(
                    f"Batch {b + 1}/{INPUTS.shape[0]//batch_size}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.6f}"
                )
                pbar.update(1)
            pbar.set_postfix_str(
                f"Epoch Loss: {epoch_loss:.6f}, Epoch Accuracy: {epoch_accuracy / (b + 1):.6f}"
            )
            print(
                "Epoch finished: %d -- Loss: %.6f -- Accuracy: %f"
                % (epoch, epoch_loss, epoch_accuracy / (b + 1))
            )
            try:
                with torch.no_grad():
                    total_test_accuracy = 0
                    for i in range(TEST_INPUTS.shape[0] // batch_size - 1):
                        test_outputs = model(
                            TEST_INPUTS[i * batch_size : (i + 1) * batch_size]
                        )
                        test_accuracy = (
                            (
                                test_outputs.argmax(dim=-1)
                                == TEST_OUTPUTS[i * batch_size : (i + 1) * batch_size]
                            )
                            .float()
                            .mean()
                        )
                        total_test_accuracy += test_accuracy
                    test_accuracy = total_test_accuracy / (i + 1)
                    print("Test Accuracy: %f" % (test_accuracy))
                    trial.report(test_accuracy, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            except:
                print("Test failed")
            toc()
    return test_accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
