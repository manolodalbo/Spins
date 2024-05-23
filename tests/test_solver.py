import unittest
from SpinTorch import spintorch
class Test_Solver(unittest.TestCase):
    def create_solver(self,batch_size):
        dx = 50e-9      # discretization (m)
        dy = 50e-9      # discretization (m)
        dz = 20e-9      # discretization (m)
        nx = 100        # size x    (cells)
        ny = 100        # size y    (cells)

        Ms = 140e3      # saturation magnetization (A/m)
        B0 = 60e-3      # bias field (T)
        Bt = 1e-3       # excitation field amplitude (T)
        learning_rate = 0.005