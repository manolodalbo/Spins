import torch
import torch.nn as nn


class MModel(nn.Module):
    def __init__(self, film, output_size=70):
        super(MModel, self).__init__()
        self.film = film
        self.output_matrix = nn.Parameter(
            torch.normal(torch.zeros(10, output_size, 2 * 16), std=0.01)
        )
        self.softamx = nn.Softmax(dim=-1)

    def forward(self, x):
        outputs = self.film(x)[:, :, 600:]
        t_per_bucket = 50
        buckets = outputs.view(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] // t_per_bucket,
            t_per_bucket,
        )
        freq_buckets = self.extract_average_frequency(signals=buckets, dt=20e-12)
        amp_buckets = buckets.sum(dim=-1)
        freq_norm = (freq_buckets - freq_buckets.mean()) / freq_buckets.std()
        amp_norm = (amp_buckets - amp_buckets.mean()) / amp_buckets.std()
        freq_and_amp = torch.cat((freq_norm, amp_norm), dim=-1)
        distance = self.distance(freq_and_amp)
        probs = self.softamx(-distance)
        return probs

    def distance(self, output):
        distances = output.unsqueeze(1) - self.output_matrix.unsqueeze(0)
        distances = (distances**2).sum(-1).sum(-1)
        return distances

    def extract_average_frequency(self, signals: torch.Tensor, dt: float):
        signal_length = signals.shape[-1]

        # Compute FFT along the last dimension
        fft_result = torch.abs(torch.fft.fft(signals, dim=-1))

        # Apply the threshold
        fft_result[fft_result < 10] = 0

        # Calculate the sampling rate
        sampling_rate = 1 / dt

        # Compute frequency values for each signal
        freq = torch.fft.fftfreq(
            signal_length, 1 / sampling_rate, device=signals.device
        )

        # Ensure freq is correctly shaped to broadcast over all preceding dimensions
        freq = freq.view(*([1] * (signals.ndim - 1)), -1)

        # Compute the weighted average of frequencies
        mult = freq * fft_result
        average = mult[..., : signal_length // 2].sum(dim=-1) / fft_result[
            ..., : signal_length // 2
        ].abs().sum(dim=-1)

        return average
