"""Damping with absorbing boundaries"""
from torch import nn, ones,log,sigmoid
from skimage.draw import rectangle_perimeter

class Damping(nn.Module):
    alpha = 1e-4    # damping coefficient ()
    alpha_max = 0.5 # maximum damping used on boundaries and for relax ()
    region_width = 10   # width of absorbing region (cells)
    alpha_min = 5e-5
    alpha_real_max = 0.55
    
    def __init__(self, dim: tuple):
        super().__init__()
        self.dim = dim
        
        A = self.alpha*ones((1, 1,) + self.dim) # damping coefficient pointwise ()
        for i in range(self.region_width):
            x, y = rectangle_perimeter((i+1, i+1), (self.dim[0]-i-2, self.dim[1]-i-2))
            A[:, :, x, y] = (1-i/self.region_width)**2*(self.alpha_max-self.alpha) + self.alpha
        Rho = -log(((self.alpha_real_max - self.alpha_min)/(A-self.alpha_min))-1)
        self.register_parameter("Rho", nn.Parameter(Rho))

    def forward(self, relax=False):
        if relax:
            return self.alpha_max
        else:
            damping_field = sigmoid(self.Rho) * (self.alpha_real_max- self.alpha_min) + self.alpha_min
            return damping_field


