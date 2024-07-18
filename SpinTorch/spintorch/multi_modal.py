import torch
import torch.nn as nn


class MModel(nn.Module):
    def __init__(self, films):
        super(MModel, self).__init__()
        self.films = nn.ModuleList(films)

    def forward(self, x):
        outputs = []
        for film in nn.ModuleList(self.films):
            unnormalized_output = film(x)
            prob = unnormalized_output[:, 1] / unnormalized_output.sum(dim=-1)
            print(prob)
            outputs.append(prob)
        combined_outputs = torch.stack(outputs, dim=1)
        return combined_outputs
