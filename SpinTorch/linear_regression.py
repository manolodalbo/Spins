import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np


class LinearReg(torch.nn.Module):
    def __init__(self):
        super(LinearReg, self).__init__()
        self.output_matrix = torch.nn.Parameter(
            torch.normal(torch.zeros((10, 81)) + 0.5, std=0.01)
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear_layer = torch.nn.Linear(81, 50)

    def forward(self, inputs):
        # lin = self.linear_layer(inputs)
        # norm = (lin - lin.mean()) / lin.std()
        distances = self.distance(inputs)
        probs = self.softmax(-distances)
        return probs

    def distance(self, output):
        distances = output.unsqueeze(1) - self.output_matrix.unsqueeze(0)
        distances = (distances**2).sum(-1)
        return distances


class LinearLayer(torch.nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(81, 24)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.second_linear = torch.nn.Linear(24, 10)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        first = self.linear(inputs)
        leaky = self.leaky_relu(first)
        second = self.second_linear(leaky)
        return self.softmax(second)


def loss_fn(preds, labels):
    epsilon = 1e-8
    log_preds = torch.log(preds + epsilon)
    to_return = torch.nn.functional.nll_loss(log_preds, labels)
    return to_return


def show_image(image: np.array):
    """
    Used to show mnist image"""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


model = LinearReg()
criteria = loss_fn
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
with open("C:\spins\data\data.p", "rb") as data_file:
    data_dict = pickle.load(data_file)
train_inputs = data_dict["train_inputs"]
test_inputs = data_dict["test_inputs"]
train_labels = data_dict["train_labels"]
test_labels = data_dict["test_labels"]
num_epochs = 2
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        optimizer.zero_grad()
        outputs = model(train_inputs[i : i + batch_size])
        loss = criteria(outputs, train_labels[i : i + batch_size])
        loss.backward()
        optimizer.step()
        accuracy = (
            (outputs.argmax(dim=-1) == train_labels[i : i + batch_size]).float().mean()
        )
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}"
        )
# calculate accuracy:
with torch.no_grad():
    model.eval()
    test_outputs = model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")
