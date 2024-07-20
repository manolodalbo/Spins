import spintorch
from spintorch.RNN_film import RNN_film
import torch


def turn_into_wave(inputs):
    inputs = inputs.transpose(-1, -2)
    to_return = fm_simpler(inputs)
    to_return = torch.flatten(to_return, start_dim=2, end_dim=-1).unsqueeze(-1)
    return to_return


def fm_simpler(inputs: torch.tensor, Fi: float = 0.5e9, Ff: float = 10e9):
    points_per_input = 600 / inputs.shape[-1]
    dt = 20e-12
    t = (
        torch.arange(0, points_per_input * dt, dt, device=inputs.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    inputs = torch.sigmoid(0.3 * inputs)  # scaled between 0 and 1
    return torch.sin(2 * torch.pi * t * (Fi) + inputs.unsqueeze(-1) * (Ff - Fi))


def main():
    Bt = 0.01
    batch_size = 64

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'

    film = RNN_film(batch_size=batch_size).to(dev)

    inputs = torch.normal(torch.zeros(64, 1, 80), std=5)
    inputs = torch.cat((turn_into_wave(inputs), torch.zeros(64, 80, 500, 1)), dim=2)
    inputs = (inputs * Bt).to(dev)
    print(inputs.shape)
    outputs = film(inputs)
    outputs = outputs.sum(dim=-1)
    print(f"mean: {outputs.squeeze().mean()}")
    print(f"std : {outputs.squeeze().std()}")


if __name__ == "__main__":
    main()
