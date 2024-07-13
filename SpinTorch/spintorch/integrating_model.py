import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class NonNegativeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NonNegativeLinear, self).__init__()
        self.weight = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # Apply ReLU to ensure weights and bias are non-negative
        non_neg_weight = F.sigmoid(self.weight)
        non_neg_bias = F.sigmoid(self.bias)
        return F.linear(x, non_neg_weight, non_neg_bias)


class IntModel(nn.Module):
    def __init__(self, film, number_of_inputs):
        super(IntModel, self).__init__()
        self.film = film
        self.linear_integrator = NonNegativeLinear(number_of_inputs, 1)

    def forward(self, x):
        film_output = self.film(x)
        plt.figure(figsize=(10, 6))
        plt.plot(film_output[0][0].cpu().detach().numpy(), label="first probe initial")
        plt.plot(film_output[0][1].cpu().detach().numpy(), label="second probe initial")
        output = self.linear_integrator(film_output)
        plt.plot(output[0][0].cpu().detach().numpy(), label="first probe")
        plt.plot(output[0][1].cpu().detach().numpy(), label="second plot")
        plt.title("Probe outputs integrated")
        plt.xlabel("Time")
        plt.ylabel("Output")
        plt.legend()
        plt.savefig("C:/spins/Spins/plots/linear_output.png")
        plt.close()
        plt.figure(figsize=(10, 6))
        weights = self.linear_integrator.weight.cpu().detach().numpy()
        plt.plot(weights[0], label="weights")
        plt.title("Weights")
        plt.legend()
        plt.savefig("C:/spins/Spins/plots/weights.png")
        plt.close()
        print(f"output shape: {output.shape}")
        return output.sum(dim=-1)
