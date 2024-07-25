import skimage
import torch


class WaveSource(torch.nn.Module):
    def __init__(self, x, y, dim=0):
        super().__init__()

        self.register_buffer("x", torch.tensor(x, dtype=torch.int64))
        self.register_buffer("y", torch.tensor(y, dtype=torch.int64))
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, B, Bt):
        Bs = B.clone()
        if Bt.shape[1] < 100:
            reproduced = Bt.unsqueeze(2).repeat(1, 1, 8).flatten(start_dim=1, end_dim=2)
            number_to_add = 100 - Bt.shape[1]
            add_first = number_to_add // 2
            add_second = number_to_add - add_first
            Bt = torch.cat(
                (
                    torch.zeros(Bt.shape[0], add_first, device=Bt.device),
                    Bt,
                    torch.zeros(Bt.shape[0], add_second, device=Bt.device),
                ),
                dim=1,
            )
        Bs[:, self.dim, self.x, self.y] = Bs[:, self.dim, self.x, self.y] + Bt
        return Bs

    def coordinates(self):
        return self.x.cpu().numpy(), self.y.cpu().numpy()


class WaveLineSource(WaveSource):
    def __init__(self, r0, c0, r1, c1, dim=0):
        x, y = skimage.draw.line(r0, c0, r1, c1)

        self.r0 = r0
        self.c0 = c0
        self.r1 = r1
        self.c1 = c1
        super().__init__(x, y, dim)
