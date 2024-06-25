import unittest
import spintorch
import numpy as np
import torch
import pickle
import spintorch.old_solver
import spintorch.old_source


class Test_Solver(unittest.TestCase):
    def create_solver(self, batch_size, Np, dt):
        dx = 50e-9  # discretization (m)
        dy = 50e-9  # discretization (m)
        dz = 20e-9  # discretization (m)
        nx = 100  # size x    (cells)
        ny = 100  # size y    (cells)

        Ms = 140e3  # saturation magnetization (A/m)
        B0 = 60e-3  # bias field (T)
        geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
        src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
        probes = []
        for p in range(Np):
            probes.append(
                spintorch.WaveIntensityProbeDisk(
                    nx - 15, int(ny * (p + 1) / (Np + 1)), 2
                )
            )
        model = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
        model.load_state_dict(
            torch.load(
                "C:/spins/Spins/models/focus_Ms/model_lowest_losssaturationMagnetization6v7.pt"
            )["model_state_dict"]
        )
        return model

    def create_old_solver(self, Np, dt):
        dx = 50e-9  # discretization (m)
        dy = 50e-9  # discretization (m)
        dz = 20e-9  # discretization (m)
        nx = 100  # size x    (cells)
        ny = 100  # size y    (cells)

        Ms = 140e3  # saturation magnetization (A/m)
        B0 = 60e-3  # bias field (T)
        geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
        src = spintorch.old_source.WaveLineSource(10, 0, 10, ny - 1, dim=2)
        probes = []
        for p in range(Np):
            probes.append(
                spintorch.WaveIntensityProbeDisk(
                    nx - 15, int(ny * (p + 1) / (Np + 1)), 2
                )
            )
        model = spintorch.old_solver.OldMMSolver(geom, dt, [src], probes)
        model_state_dict = torch.load(
            "C:/spins/Spins/models/focus_Ms/model_lowest_losssaturationMagnetization6v7.pt"
        )["model_state_dict"]
        model_state_dict["m0"] = model_state_dict["m0"][0].unsqueeze(0)
        model.load_state_dict(model_state_dict)
        return model

    # def test_model(self):
    #     """Essentially just tests that different batch_sizes"""
    #     dev = torch.device("cpu")
    #     dt = 20e-12
    #     model_single = self.create_solver(batch_size=1, Np=3, dt=dt)
    #     model_single.to(dev)
    #     model_multiple = self.create_solver(batch_size=3, Np=3, dt=dt)
    #     model_multiple.to(dev)
    #     Bt = 1e-3
    #     timesteps = 600
    #     t = torch.arange(0, timesteps * dt, dt).unsqueeze(0).unsqueeze(2)
    #     single_input = Bt * torch.sin(2 * np.pi * 4e9 * t).to(dev)
    #     X1 = single_input
    #     X2 = Bt * torch.sin(2 * np.pi * 3.5e9 * t)
    #     X3 = Bt * torch.sin(2 * np.pi * 4.5e9 * t)
    #     multiple_inputs = torch.cat((X1, X2, X3), dim=0).to(dev)
    #     single_output = model_single(single_input)
    #     multiple_output = model_multiple(multiple_inputs)
    #     self.assertTrue(torch.allclose(single_output[0], multiple_output[0]))

    def test_multiple_batches(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = 20e-12
        old_model = self.create_old_solver(Np=2, dt=dt)
        new_model = self.create_solver(batch_size=64, Np=2, dt=dt)
        old_model.to(dev)
        new_model.to(dev)
        Bt = 0.01
        with open(f"C:\spins\data\data.p", "rb") as data_file:
            data_dict = pickle.load(data_file)
        TEST_INPUTS = (
            torch.tensor(data_dict["test_inputs"] * Bt)[0:64].unsqueeze(-1).to(dev)
        )
        print(TEST_INPUTS.shape)
        output_new = new_model(TEST_INPUTS)
        output_old = old_model(TEST_INPUTS[7].unsqueeze(0)).sum(dim=1)
        print(output_new[7])
        print(output_old[0])
        assert torch.allclose(output_new[7], output_old[0])


if __name__ == "__main__":
    unittest.main()
