import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class IModel(nn.Module):
    def __init__(self, film, input_size, end_first):
        super(IModel, self).__init__()
        self.film = film
        self.end_first = end_first
        self.input_size = input_size
        self.t_per_bucket = 20
        self.integrator = nn.Parameter(
            torch.normal(
                torch.zeros(1, 1, (input_size - end_first) // self.t_per_bucket),
                std=0.01,
            )
        )

    def forward(self, inputs):
        outputs = self.film(inputs)
        # plt.figure()
        # plt.plot(outputs[0, 0, :].detach().cpu().numpy())
        # plt.plot(outputs[0, 1, :].detach().cpu().numpy())

        # plt.savefig("one.png")
        # plt.close()
        # plt.figure()
        # plt.plot(outputs[1, 0, :].squeeze().detach().cpu().numpy())
        # plt.plot(outputs[1, 1, :].squeeze().detach().cpu().numpy())
        # plt.savefig("two.png")
        # plt.close()
        outputs = outputs[:, :, self.end_first :]
        buckets = outputs.view(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] // self.t_per_bucket,
            self.t_per_bucket,
        )
        amps = buckets.sum(dim=-1)
        # mult = amps * nn.functional.sigmoid(self.integrator)
        intensities = amps.sum(dim=-1)
        probs = intensities / intensities.sum(dim=-1).unsqueeze(-1)
        return probs
