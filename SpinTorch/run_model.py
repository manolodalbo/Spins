import torch
import spintorch
import pickle

"""Parameters"""
dx = 50e-9  # discretization (m)
dy = 50e-9  # discretization (m)
dz = 20e-9  # discretization (m)
nx = 100  # size x    (cells)
ny = 100  # size y    (cells)

Ms = 140e3  # saturation magnetization (A/m)
B0 = 60e-3  # bias field (T)
Bt = 1e-3  # excitation field amplitude (T)

dt = 20e-12  # timestep (s)
batch_size = 64
"""Directories"""
basedir = "focus_Ms/"
plotdir = "plots/" + basedir

"""Geometry, sources, probes, model definitions"""
### Here are three geometry modules initialized, just uncomment one of them to try:
# Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
# r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
# rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
# rho = torch.zeros((rx, ry))  # Design parameter array
# geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0,
#                                     r0, dr, dm, z_off, rx, ry, Ms_CoPt)
B1 = 50e-3  # training field multiplier (T)
geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
# geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
probes = []
Np = 2  # number of probes
for p in range(Np):
    probes.append(
        spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / (Np + 1)), 2)
    )
model = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
dev_name = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(dev_name)  # 'cuda' or 'cpu'
print("Running on", dev)
model.load_state_dict(
    torch.load("C:/spins/Spins/models/focus_Ms/model_e13nine_low_points_low_lr.pt")[
        "model_state_dict"
    ]
)
model.to(dev)  # sending model to GPU/CPU
with open("C:\spins\data\data.p", "rb") as data_file:
    data_dict = pickle.load(data_file)
TEST_INPUTS = torch.tensor(data_dict["test_inputs"] * Bt).unsqueeze(-1).to(dev)
print(TEST_INPUTS.shape)
TEST_LABELS = data_dict["test_labels"].to(dev)
with torch.no_grad():
    total_test_accuracy = 0
    for i in range(TEST_INPUTS.shape[0] // 64 - 1):
        test_outputs = model(TEST_INPUTS[i * 64 : (i + 1) * 64])
        test_accuracy = (
            (test_outputs.argmax(dim=-1) == TEST_LABELS[i * 64 : (i + 1) * 64])
            .float()
            .mean()
        )
        total_test_accuracy += test_accuracy
    test_accuracy = total_test_accuracy / (i + 1)
    print("Test Accuracy: %f" % (test_accuracy))
