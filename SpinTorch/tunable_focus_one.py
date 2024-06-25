"""Optimize a focusing model"""

import torch
import os
import spintorch
from spintorch.utils import tic
import optuna
from optuna.trial import TrialState
import tunable_preprocess


def objective(trial):
    """Parameters"""
    dx = 50e-9  # discretization (m)
    dy = 50e-9  # discretization (m)
    dz = 20e-9  # discretization (m)
    nx = 100  # size x    (cells)
    ny = 100  # size y    (cells)

    Ms = 140e3  # saturation magnetization (A/m)
    B0 = 60e-3  # bias field (T)
    # dt = 1 / (1600 * 3e6)  # timestep (s)
    dt = trial.suggest_float("dt", 2e-12, 30e-12)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    B1 = 50e-3  # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    # geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = []
    Np = 3  # number of probes
    for p in range(Np):
        probes.append(
            spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
        )
    film = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    Bt = trial.suggest_float("Bt", 0.005, 0.1)  # excitation field amplitude (T)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5)
    epochs = 200
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
    middle_size = trial.suggest_int("middle_size", 50, 3000)
    data_dict = tunable_preprocess.preprocess(middle_size)
    INPUTS = (data_dict["signals"] * Bt).float().unsqueeze(-1).to(dev)
    OUTPUTS = data_dict["train_labels"]  # all classes in outputs
    print(f"Inputs shape: {INPUTS.shape}")
    OUTPUTS = OUTPUTS.to(dev)
    TEST_INPUTS = (data_dict["test_signals"] * Bt).unsqueeze(-1).to(dev)
    TEST_OUTPUTS = data_dict["test_labels"].to(dev)  # desired output
    """Define optimizer and lossfunction"""
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=learning_rate
    )
    """Train the network"""
    print(INPUTS.shape)
    tic()
    model.retain_history = False

    def loss_func(output, target_index):
        print(output)
        print(target_index)
        output = output / output.sum(dim=-1).unsqueeze(-1)
        print(f"output: {output.shape}, target: {target_index.shape}")
        return torch.nn.functional.cross_entropy(output, target_index)

    for epoch in range(0, epochs):
        indices = torch.randperm(INPUTS.shape[0], device=dev)
        INPUTS = INPUTS[indices]
        OUTPUTS = OUTPUTS[indices]
        epoch_loss = 0
        epoch_accuracy = 0
        for b, b1 in enumerate(range(batch_size, INPUTS.shape[0] + 1, batch_size)):
            b0 = b1 - batch_size
            u = model(INPUTS[b0:b1])
            loss = loss_func(u, OUTPUTS[b0:b1])
            epoch_loss += loss.item()
            accuracy = (u.argmax(dim=-1) == OUTPUTS[b0:b1]).float().mean()
            epoch_accuracy += accuracy
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_positives = 0
            total_positive_accurate = 0
            total_test_accuracy = 0.0

            num_batches = TEST_INPUTS.shape[0] // batch_size
            for i in range(num_batches):
                test_batch_inputs = TEST_INPUTS[i * batch_size : (i + 1) * batch_size]
                test_batch_outputs = TEST_OUTPUTS[i * batch_size : (i + 1) * batch_size]

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
            trial.report(avg_test_accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return avg_test_accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
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
