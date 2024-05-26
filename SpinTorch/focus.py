"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
import matplotlib.pyplot as plt
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

def fm(inputs: np.array, Fi: float, Ff: float) -> np.array:
    """
    Frequency modulate the input images.
    
    Parameters:
        inputs (np.array): Array of shape (number of inputs, 784), where each row represents an image.
        Fi (float): Minimum frequency in Hz.
        Ff (float): Final frequency in Hz.
    
    Returns:
        np.array: Frequency modulated waveforms for each input image.
    """
    points_per_input = 1
    dt = 20e-12     # timestep (s)
    timesteps = inputs.shape[1] * points_per_input
    t = np.arange(0, timesteps * dt, dt)  # time vector
    modulated_wave = np.zeros((inputs.shape[0], timesteps),dtype="float32")
    pbar = tqdm(inputs)
    for i in range(inputs.shape[0]):
        pbar.set_description(f"[({i+1}/{len(inputs)})] Processing images into waves")
        pos_deriv = True
        prev = 0
        for j, pixel_intensity in enumerate(inputs[i]):
            # Calculate the corresponding frequency for this pixel
            frequency = Fi + pixel_intensity * (Ff - Fi)
            if j>0:
                phase = np.arcsin(prev)
                if not pos_deriv:
                    phase = np.pi - phase
                modulated_wave[i, points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (1.5))*np.sin(2 * np.pi * frequency * t[1:points_per_input+1] + phase)
                if np.cos(2*np.pi*frequency*t[points_per_input] + phase) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency * t[3] + phase)
            else:
                modulated_wave[i,points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (1.5))*np.sin(2*np.pi*frequency*t[0:points_per_input])
                if np.cos(2*np.pi*frequency*t[points_per_input-1]) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency *t[2])
    return modulated_wave
"""Parameters"""
dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 100        # size x    (cells)
ny = 100        # size y    (cells)

Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
Bt = 1e-3       # excitation field amplitude (T)

dt = 20e-12     # timestep (s)
f1 = 4e9        # source frequency (Hz)
f2 = 3.5e9
f3 = 3e9
timesteps = 600 # number of timesteps for wave propagation
learning_rate = 0.001
batch_size = 2


'''Directories'''
basedir = 'focus_Ms/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
### Here are three geometry modules initialized, just uncomment one of them to try:
# Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
# r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
# rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
# rho = torch.zeros((rx, ry))  # Design parameter array
# geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, 
#                                     r0, dr, dm, z_off, rx, ry, Ms_CoPt)
# B1 = 50e-3      # training field multiplier (T)
# geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
probes = []
epochs = 20
Np = 2  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
model = spintorch.MMSolver(geom, dt, batch_size, [src], probes)

dev = torch.device('cpu')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU

with open('C:\spin\data\data.p','rb') as data_file:
    data_dict = pickle.load(data_file)
high_then_low = torch.cat((torch.ones(300,1),torch.zeros(300,1)))
low_then_high = torch.cat((torch.zeros(300,1),torch.ones(300,1)))
INPUTS = torch.tensor(Bt * fm(np.array([high_then_low,low_then_high]),3e9,5e9)).unsqueeze(-1).to(dev)
#INPUTS = torch.tensor(data_dict['train_inputs']*Bt).unsqueeze(-1).to(dev)
OUTPUTS = torch.tensor([0,1],dtype=torch.long).to(dev) # desired output

'''Define optimizer and lossfunction'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def my_loss(output, target_index):
    output = output/(output.sum(dim=-1).unsqueeze(-1))
    return torch.nn.functional.cross_entropy(output,target_index)
'''Load checkpoint'''
epoch = epoch_init = -1 # select previous checkpoint (-1 = don't use checkpoint)
if epoch_init>=0:
    checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []

'''Train the network'''
print(INPUTS.shape)
pbar = tqdm(total=INPUTS.shape[0]//batch_size)
tic()
model.retain_history = True
for epoch in range(epoch_init+1, epochs):
    u = model(INPUTS)
    print(u)
    loss = my_loss(u,OUTPUTS)
    stat_cuda('after forward')
    loss.backward()
    optimizer.step()
    stat_cuda('after backward')
    loss_iter.append(loss.item())  # store loss values
    spintorch.plot.plot_loss(loss_iter, plotdir)
    print("Epoch finished: %d -- Loss: %.6f" % (epoch, loss))
    toc()   

    '''Save model checkpoint'''
    torch.save({
                'epoch': epoch,
                'loss_iter': loss_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savedir + 'model_e%d.pt' % (epoch))
    
    '''Plot spin-wave propagation'''
    if model.retain_history:
        with torch.no_grad():
            spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir)
            mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
            wave_snapshot(model, mz[timesteps-1], (plotdir+'snapshot_time%d_epoch%d.png' % (timesteps,epoch)),r"$m_z$")
            wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+'snapshot_time%d_epoch%d.png' % (int(timesteps/2),epoch)),r"$m_z$")
            wave_integrated(model, mz, (plotdir+'integrated_epoch%d.png' % (epoch)))


  
