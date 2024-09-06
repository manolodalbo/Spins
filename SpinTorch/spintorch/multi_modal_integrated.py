import torch
import torch.nn as nn


class IModel(nn.Module):
    def __init__(self, film, output_size=70):
        super(IModel, self).__init__()
        self.film = film
        self.output_matrix = nn.Parameter(
            torch.normal(torch.zeros(10, output_size), std=0.01)
        )
        self.softamx = nn.Softmax(dim=-1)

    def forward(self, x):
        outputs = self.film(x)[:, :, 600:].sum(dim=-1)
        outputs = (outputs - outputs.mean()) / outputs.std()
        print(outputs.shape)
        distance = self.distance(outputs)
        print("distances shape:")
        print(distance.shape)
        probs = self.softamx(-distance)
        return probs

    def distance(self, output):
        distances = output.unsqueeze(1) - self.output_matrix.unsqueeze(0)
        distances = (distances**2).sum(-1)
        return distances
