import torch
import pickle


class LinearLayer(torch.nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(81, 10)

    def forward(self, inputs):
        return self.linear(inputs)


model = LinearLayer()
criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
with open("C:\spins\data\data.p", "rb") as data_file:
    data_dict = pickle.load(data_file)
train_inputs = torch.tensor(data_dict["dig_train_inputs"])
print(train_inputs.shape)
test_inputs = torch.tensor(data_dict["dig_test_inputs"])
train_labels = data_dict["train_labels"]
test_labels = data_dict["test_labels"]
num_epochs = 300


def cross_entropy(outputs, target_index):
    # target_index = target_index.long()
    # ohe = torch.nn.functional.one_hot(target_index, 2).float()
    preds = outputs / (outputs.sum(dim=-1).unsqueeze(-1))
    loss = torch.nn.functional.cross_entropy(preds, target_index)
    return loss


for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = torch.nn.functional.cross_entropy(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
# calculate accuracy:
with torch.no_grad():
    model.eval()
    test_outputs = model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")
