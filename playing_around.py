import torch
import matplotlib.pyplot as plt
from preprocess import fm
# Number of timesteps
n_timesteps = 600
# Generate 50 equally spaced values from 0 to 2*pi
x = torch.linspace(0, 2 * torch.pi, n_timesteps)

# Compute the sine of these values
y = torch.sin(x)
plt.figure()
plt.plot(y)
plt.xlabel("timesteps")
plt.title("sine wave")
plt.show()
y = ((y + 1)/2).unsqueeze(0)
# Print the values
min_freq = 0.5e9
max_freq = 10e9
output = fm(y,Fi=min_freq,Ff = max_freq,samples_per_point=1)
plt.figure()
plt.plot(output[0])
plt.title("input wave")
plt.xlabel("timesteps(dt = 20e-12s)")
plt.show()
exit()
outputs = ((max_freq - min_freq) / 2) * y + ((max_freq + min_freq) / 2)
print(outputs.shape)
dt = 20e-12
t = torch.arange(0, 600 * dt, dt).unsqueeze(0)  # time vector
print(t.shape)
inside = 2 * torch.pi * outputs.unsqueeze(-1) * t
print(inside.shape)
inputs = torch.sin(inside).unsqueeze(-1)
print(inputs.shape)
