import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch

n_timesteps = 10000
# Generate 50 equally spaced values from 0 to 2*pi
x = torch.linspace(0, 2 * torch.pi, n_timesteps)

# Compute the sine of these values
y = torch.sin(x)

# Print the values
min_freq = 3e7
max_freq = 30e7

outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
print(outputs.shape)
dt = 20e-12
t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
inside = 2 * torch.pi * outputs.unsqueeze(-1) * t
tensor = torch.sin(inside)
print(tensor.shape)
x, y = np.meshgrid(np.arange(tensor.shape[-1]), np.linspace(0, 80, tensor.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(x, y, np.array(tensor), cmap="viridis")
ax.set_xlabel("t")
ax.set_ylabel("width of film")
ax.set_zlabel("Intensity")
plt.show()
