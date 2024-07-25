from preprocess import get_data, turn_into_wave
import numpy as np
import torch
import torch.nn as nn
from SpinTorch.spintorch.RNN_film import RNN_film
import SpinTorch.spintorch as spintorch
import os


class MyTrigram(nn.Module):
    def __init__(self, vocab_size, batch_size, embed_size=10, output_size=50, Bt=0.01):
        super(MyTrigram, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.linear_layer = nn.Linear(2 * self.embed_size, 50)
        self.activation = nn.LeakyReLU()
        self.second_layer = nn.Linear(50, self.vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.embedding_matrix = nn.Parameter(
            torch.normal(torch.zeros(self.vocab_size, self.embed_size), std=0.01),
        )
        self.output_matrix = nn.Parameter(
            torch.normal(torch.zeros(self.vocab_size, self.output_size), std=0.01),
        )

    def forward(self, inputs):
        full_input = self.embedding_matrix[inputs.int()]
        flattened = full_input.flatten(start_dim=1, end_dim=2)
        first = self.linear_layer(flattened)
        activated = self.activation(first)
        second_layer_output = self.second_layer(activated)
        # sigmoid = self.sigmoid(second_layer_output)
        probs = self.softmax(second_layer_output)
        # first = (first - first.mean()) / first.std()
        # distance = self.euclidean_distance(first)
        # probs = self.softmax(-distance)
        return probs

    def euclidean_distance(self, outputs):
        """ "
        Note: not actually euclidean distance because I don't take the square root after adding
        Should takes in the the output after passing through film and normalized(shape batch_size x embed_size) and compares
        with the embedding matrix. Returns batch_size x vocab_size
        """
        distances = outputs.unsqueeze(1) - self.output_matrix.unsqueeze(
            0
        )  # bs X vocab_size X embed_size
        distances = (distances**2).sum(
            dim=-1
        )  # essentially gets mean squared error(this may not be necessary)

        # there are two ways to do this:
        # the first is to use the distances to compute logits and the second is to directly
        # use the mean squared error to compute the loss, however, the embeddings would have
        # to be fixed because or else all the embeddings could be the same and would still result in 0 loss every time
        # i dont think MSE will work in this case without computing probs later
        # useful bc it gets rid of negatives and punishes terrible guesses
        square_root = torch.sqrt(distances)
        return square_root


def loss_fn(preds, labels):
    epsilon = 1e-8
    log_preds = torch.log(preds + epsilon)
    to_return = torch.nn.functional.nll_loss(log_preds, labels)
    return to_return


def perplexity(labels, preds):
    """
    Compute the perplexity of predictions.
    :param labels: ground truth labels
    :param preds: predictions
    :return: perplexity value
    """
    log_preds = torch.log(preds + 1e-8)
    loss = torch.nn.functional.nll_loss(log_preds, labels)
    return torch.exp(torch.mean(loss))


def main():
    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    epochs = 10
    batch_size = 128
    embed_size = 80
    learning_rate = 0.01

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'

    data_path = "../data"
    data_path = "C:/spin/data"
    train_tokens, test_tokens, vocab = get_data(
        f"{data_path}/train.txt", f"{data_path}/test.txt"
    )
    train_array = np.array(train_tokens)
    test_array = np.array(test_tokens)
    X0, Y0 = np.vstack([train_array[0:-2], train_array[1:-1]]).T, train_array[2:]
    X1, Y1 = np.vstack([test_array[0:-2], test_array[1:-1]]).T, test_array[2:]
    model = MyTrigram(len(vocab), batch_size, embed_size=embed_size).to(dev)
    criterion = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_iter = []
    perplexity_iter = []
    to_print = []
    for epoch in range(epochs):
        perplexity_running_avg = 0
        loss_running_avg = 0
        for i in range(0, len(X0), batch_size):
            inputs = torch.tensor(X0[i : i + batch_size], dtype=torch.float32).to(dev)
            targets = torch.tensor(Y0[i : i + batch_size], dtype=torch.long).to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            perp = perplexity(targets, outputs)
            loss_running_avg = loss_running_avg + (loss.item() - loss_running_avg) / (
                i + 1
            )
            perplexity_running_avg = perplexity_running_avg + (
                perp.item() - perplexity_running_avg
            ) / (i + 1)
            loss_iter.append(loss_running_avg)
            perplexity_iter.append(perplexity_running_avg)
            if epoch < 1 and i // batch_size < 100:
                to_print.append(loss.item())
            else:
                spintorch.plot.plot_loss(
                    np.array(to_print), plotdir, "loss_first_50_normal_linear_simple",xlabel="Batch",title="Normal Digital Trigram"
                )
                spintorch.plot.plot_loss(
                    np.array(loss_iter),plotdir,"average_loss_linear_trigram", xlabel="Batch", title="Digital Trigram Running Average"
                )
                exit()

        print(
            "Epoch finished: perplexity: ", perplexity_iter[-1], "loss: ", loss_iter[-1]
        )
        spintorch.plot.plot_loss(
            np.array(loss_iter), plotdir, "linear_trigram_seperatematrix"
        )
        spintorch.plot.plot_loss(
            np.array(perplexity_iter),
            plotdir,
            "linear_trigram_perplexity_seperatematrix",
        )


if __name__ == "__main__":
    main()
