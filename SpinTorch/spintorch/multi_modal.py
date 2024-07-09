import torch.nn as nn
import matplotlib.pyplot as plt
import torch


class MModel(nn.Module):
    def __init__(self, ifilm, cfilm):
        super(MModel, self).__init__()
        self.films = nn.ModuleList([ifilm, cfilm])

    def forward(self, x):
        ifilm = self.films[0]
        cfilm = self.films[1]
        ifilm_output = ifilm(x)
        small_value = 1e-12
        ifilm_output = torch.where(
            ifilm_output == 0, torch.tensor(small_value), ifilm_output
        )
        cfilm_output = cfilm(x)
        plt.figure(figsize=(10, 6))
        plt.plot(ifilm_output[0][0].cpu().detach().numpy(), label="first probe")
        plt.plot(ifilm_output[0][1].cpu().detach().numpy(), label="second probe")

        # Adding titles and labels
        plt.title("Tensors Plot")
        plt.xlabel("Time")
        plt.ylabel("Output")
        plt.legend()

        # # Show plot
        plt.savefig("C:/spins/Spins/plots/tensors_plot_zeros.png")

        # plt.figure(figsize=(10, 6))
        # plt.plot(x[0].cpu().detach().numpy(), label="input")
        # plt.title("Input Plot")
        # plt.xlabel("Time")
        # plt.ylabel("Input")
        # plt.legend()
        # plt.savefig("C:/spins/Spins/plots/input_plot.png")
        integration = ifilm_output[:, 0, :] / ifilm_output.sum(dim=1)
        output = cfilm_output * integration.unsqueeze(1)
        return output.sum(dim=-1)
