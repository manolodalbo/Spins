import torch.nn as nn


class OldModel(nn.Module):
    def __init__(self, film, output_size=70):
        super(OldModel, self).__init__()
        self.film = film
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        outputs = self.film(x)[:, :, 600:].sum(dim=-1)
        outputs = (outputs - outputs.mean()) / outputs.std()
        probs = self.softmax(outputs)
        return probs
