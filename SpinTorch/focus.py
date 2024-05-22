"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")


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
learning_rate = 0.005


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
batch_size = 3
geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
probes = []
epochs = 20
Np = 3  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
model = spintorch.MMSolver(geom, dt, 3, [src], probes)

dev = torch.device('cpu')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal and output goal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X1 = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude
X2 = Bt*torch.sin(2*np.pi*f2*t)
X3 = Bt*torch.sin(2*np.pi*f3*t)
INPUTS = torch.cat((X1,X2,X3),dim=0)  # here we could cat multiple inputs
# INPUTS = Bt*torch.sin(2*np.pi*f1*t) # here we could cat multiple inputs
OUTPUTS = torch.tensor(np.array([0,1,2]),dtype=torch.long).to(dev) # desired output

'''Define optimizer and lossfunction'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def my_loss(output, target_index):
    print(output)
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
tic()
model.retain_history = True
for epoch in range(epoch_init+1, epochs):
    optimizer.zero_grad()
    u = model(INPUTS)
    print("output: ")
    print(u)
    loss = my_loss(u,OUTPUTS)
    loss_iter.append(loss.item())  # store loss values
    spintorch.plot.plot_loss(loss_iter, plotdir)
    stat_cuda('after forward')
    loss.backward()
    optimizer.step()
    stat_cuda('after backward')
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


  
