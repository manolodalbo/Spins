import torch
import torch.nn as nn


class TModel(nn.Module):
    def __init__(self, film, input_size, end_first):
        super(TModel, self).__init__()
        self.film = film
        self.output_matrix = nn.Parameter(
            torch.normal(torch.zeros(2, (input_size - end_first) // 10), std=0.01)
        ).to("cuda")
        self.input_size = input_size
        self.end_first = end_first
        self.number_of_buckets = (input_size - end_first) // 20
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = self.film(inputs)
        bucketed_output = self.bucket(output[:, :, self.end_first :])
        distances = self.euclidean_distance(bucketed_output.flatten(2)).squeeze()
        probs = nn.functional.softmax(distances, dim=-1)
        return probs

    def bucket(self, outputs, t_per_bucket=20):
        buckets = outputs.view(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] // t_per_bucket,
            t_per_bucket,
        )
        freq_buckets = extract_average_frequency(buckets, 20e-12)
        freq_norm = (freq_buckets - freq_buckets.mean()) / freq_buckets.std()
        amp_buckets = buckets.sum(dim=-1)
        amp_normalized = (amp_buckets - amp_buckets.mean()) / amp_buckets.std()
        full_output = torch.cat(
            [amp_normalized.squeeze(0).unsqueeze(2), freq_norm.squeeze(0).unsqueeze(2)],
            dim=2,
        )
        return full_output

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


def extract_average_frequency(signals: torch.Tensor, dt: float):
    signal_length = signals.shape[-1]

    # Compute FFT along the last dimension
    fft_result = torch.abs(torch.fft.fft(signals, dim=-1))

    # Apply the threshold
    fft_result[fft_result < 10] = 0

    # Calculate the sampling rate
    sampling_rate = 1 / dt

    # Compute frequency values for each signal
    freq = torch.fft.fftfreq(signal_length, 1 / sampling_rate, device=signals.device)

    # Ensure freq is correctly shaped to broadcast over all preceding dimensions
    freq = freq.view(*([1] * (signals.ndim - 1)), -1)

    # Compute the weighted average of frequencies
    mult = freq * fft_result
    average = mult[..., : signal_length // 2].sum(dim=-1) / fft_result[
        ..., : signal_length // 2
    ].abs().sum(dim=-1)

    return average
