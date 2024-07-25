from preprocess import get_data, turn_into_wave
import numpy as np
import torch
import torch.nn as nn
from SpinTorch.spintorch.RNN_film import RNN_film
import SpinTorch.spintorch as spintorch
import os
import matplotlib.pyplot as plt


class MyTrigram(nn.Module):
    def __init__(self, vocab_size, batch_size, embed_size=10, Bt=0.01):
        super(MyTrigram, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.Bt = Bt
        self.film_RNN = RNN_film(
            self.embed_size, batch_size=self.batch_size, output_size=80
        )
        self.embedding_matrix = nn.Parameter(
            torch.normal(torch.zeros(self.vocab_size, self.embed_size), std=0.01)
        )
        # self.output_matrix = nn.Parameter(
        #     torch.normal(torch.ones(self.vocab_size, self.embed_size) * 7.8e6, std=1e5)
        # )
        self.output_matrix = nn.Parameter(
            torch.normal(torch.zeros(self.vocab_size, self.embed_size), std=0.01)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        wave_inputs = turn_into_wave(inputs, self.embedding_matrix)
        full_input = torch.cat(
            [
                wave_inputs,
                torch.zeros(
                    (wave_inputs.shape[0], wave_inputs.shape[1], 500, 1),
                    device=wave_inputs.device,
                ),
            ],
            dim=2,
        )
        film_output = self.film_RNN(full_input * self.Bt)
        # it might be worthwhile to crop the output to only once both inputs have gone
        # it might also make sense to just weight later outputs more strongly. linearly increasing weights?
        film_output = film_output.sum(dim=-1)
        print(f"ouput mean: {film_output.mean()}  output std: {film_output.std()}")
        film_output = (film_output - film_output.mean()) * 5 / film_output.std()
        print(
            f"norm ouput mean: {film_output.mean()} norm output std: {film_output.std()}"
        )

        distances = self.euclidean_distance(film_output)  # batch_size x vocab_size
        if distances.isnan().any():
            print("distances are none")
            exit()
        normalized_distances = (distances - distances.mean()) / distances.std()
        if normalized_distances.isnan().any():
            print("normalized distances are nan")
            exit()
        probs = self.softmax(-distances)
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
    print(preds[0])
    epsilon = 1e-8
    log_preds = torch.log(preds + epsilon)
    to_return = torch.nn.functional.nll_loss(log_preds, labels)
    return to_return


def perplexity(preds, labels):
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
    for epoch in range(epochs):
        for i in range(0, len(X0), batch_size):
            inputs = torch.tensor(X0[i : i + batch_size], dtype=torch.float32).to(dev)
            targets = torch.tensor(Y0[i : i + batch_size], dtype=torch.long).to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            perp = perplexity(outputs, targets)
            loss_iter.append(loss.item())
            spintorch.plot.plot_loss(
                loss_iter, plotdir, "new_loss_output_normalizedstd5lr0.01"
            )
            print(f"Epoch {epoch} Batch {i} Loss: {loss} Perplexity: {perp}")
        print("Epoch finished: perplexity: ", perp)


if __name__ == "__main__":
    main()
