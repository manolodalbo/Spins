import unittest
import spintorch
import numpy as np
import torch
class Test_Solver(unittest.TestCase):
    def create_solver(self,batch_size,Np,dt):
        dx = 50e-9      # discretization (m)
        dy = 50e-9      # discretization (m)
        dz = 20e-9      # discretization (m)
        nx = 100        # size x    (cells)
        ny = 100        # size y    (cells)

        Ms = 140e3      # saturation magnetization (A/m)
        B0 = 60e-3      # bias field (T)
        geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
        src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
        probes = []
        for p in range(Np):
            probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
        model = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
        return model
    def test_model(self):
        """ Essentially just tests that different batch_sizes"""
        dev = torch.device('cpu')
        dt = 20e-12
        model_single = self.create_solver(batch_size=1,Np=3,dt=dt)
        model_single.to(dev)
        model_multiple = self.create_solver(batch_size=3,Np = 3,dt=dt)
        model_multiple.to(dev)
        Bt = 1e-3
        timesteps = 600
        t = torch.arange(0,timesteps*dt,dt).unsqueeze(0).unsqueeze(2)
        single_input = Bt * torch.sin(2*np.pi *4e9*t).to(dev)
        X1 = single_input
        X2 = Bt * torch.sin(2*np.pi *3.5e9*t)
        X3 = Bt * torch.sin(2*np.pi*4.5e9*t)
        multiple_inputs = torch.cat((X1,X2,X3),dim=0).to(dev)
        single_output = model_single(single_input)
        multiple_output = model_multiple(multiple_inputs)
        self.assertTrue(torch.allclose(single_output[0],multiple_output[0]))
if __name__ == '__main__':
    unittest.main()
