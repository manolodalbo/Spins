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
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--plot_name', type=str, default='')
    parser.add_argument('--Bt', type=float, default=1e-3)
    args = parser.parse_args()
    return args
def focus(args):
        
    """Parameters"""
    dx = 50e-9      # discretization (m)
    dy = 50e-9      # discretization (m)
    dz = 20e-9      # discretization (m)
    nx = 100        # size x    (cells)
    ny = 100        # size y    (cells)

    Ms = 140e3      # saturation magnetization (A/m)
    B0 = 60e-3      # bias field (T)
    Bt = args.Bt       # excitation field amplitude (T)

    dt = 20e-12     # timestep (s)
    learning_rate = args.learning_rate
    batch_size = args.batch_size
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
    B1 = 50e-3      # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
    # geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
    src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
    probes = []
    epochs = args.epochs
    Np = 2  # number of probes
    for p in range(Np):
        probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
    model = spintorch.MMSolver(geom, dt, batch_size, [src], probes)
    dev_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'
    print('Running on', dev)
    model.to(dev)   # sending model to GPU/CPU

    with open('C:\spins\data\data.p','rb') as data_file:
        data_dict = pickle.load(data_file)
    INPUTS = torch.tensor(data_dict['train_inputs']*Bt).unsqueeze(-1).to(dev)
    print("inputs shape: ")
    print(INPUTS.shape)
    OUTPUTS = torch.tensor(data_dict['train_labels'],dtype=torch.long).to(dev) # desired output
    print(OUTPUTS)
    '''Define optimizer and lossfunction'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    def ohe(target_values: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        One-hot encodes the target values.
        
        Parameters:
            target_values (torch.Tensor): Tensor of shape (batch size, 1) containing the target values.
            num_classes (int): Number of classes.
        
        Returns:
            torch.Tensor: One-hot encoded tensor of shape (batch size, num_classes).
        """
        return torch.eye(num_classes,device=dev)[target_values]
    def my_cce(output, target_index):
        average = output.sum(dim=-1).unsqueeze(-1)/output.size()[-1]
        output = output - average
        encoded_targets = ohe(target_index, output.size()[-1])
        preds = output/output.abs().sum(dim=-1).unsqueeze(-1)
        print("initial preds")
        print(preds)
        preds = torch.nn.functional.softmax(preds,dim=-1)
        print("softmax preds")
        print(preds)
        logged = -torch.log(preds)*encoded_targets
        print(logged)
        cce = torch.sum(logged,dim=-1)
        return cce.mean()
    def log_loss_norm(output,target_index):
        encoded_targets = ohe(target_index, output.size()[-1])
        preds = output/output.sum(dim=-1).unsqueeze(-1)
        logged = -torch.log(preds)*encoded_targets
        cce = torch.sum(logged,dim=-1)
        return cce.mean()
    def their_loss(output, target_index):
        output = output/output.sum(dim=-1).unsqueeze(-1)
        print(output)
        return torch.nn.functional.cross_entropy(output,target_index)
    def loss_combo(output,target_index):
        average = output.sum(dim=-1).unsqueeze(-1)/output.size()[-1]
        output = output - average
        preds = output/output.abs().sum(dim=-1).unsqueeze(-1)
        loss = torch.nn.functional.cross_entropy(preds,target_index)
        return loss
    def div_loss(output,target_index):
        output = output/10e13
        loss = torch.nn.functional.cross_entropy(output,target_index)
        return loss
    def bce(output,target_index):
        ohe = torch.nn.functional.one_hot(target_index,2).float()
        preds = output/(output.sum(dim=-1).unsqueeze(-1))
        loss = torch.nn.functional.binary_cross_entropy(preds,ohe)
        return loss
    def bce_avg(output,target_index):
        ohe = torch.nn.functional.one_hot(target_index,2).float()
        average = output.sum(dim=-1).unsqueeze(-1)/output.size()[-1]
        output = output - average
        preds = output/output.abs().sum(dim=-1).unsqueeze(-1)
        print(preds)
        loss = torch.nn.functional.binary_cross_entropy(preds,ohe)
        return loss
    def return_loss(loss: str):
        mapping = {'bce':bce,'my_cce':my_cce,'log_loss_norm':log_loss_norm,'their_loss':their_loss,'loss_combo':loss_combo,'div_loss':div_loss}
        return mapping.get(loss,log_loss_norm)
    loss_func = return_loss(args.loss)


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
        for b,b1 in enumerate(range(batch_size,INPUTS.shape[0]+1,batch_size)):
            b0 = b1 - batch_size
            u = model(INPUTS[b0:b1])
            loss = loss_func(u,OUTPUTS[b0:b1])
            accuracy = (u.argmax(dim=-1)==OUTPUTS).float().mean()
            stat_cuda('after forward')
            loss.backward()
            optimizer.step()
            stat_cuda('after backward')
            pbar.set_description(f'Batch {b + 1}/{INPUTS.shape[0]//batch_size}, Loss: {loss.item()}')
            pbar.update(1)
        loss_iter.append(loss.item())  # store loss values
        try:
            spintorch.plot.plot_loss(loss_iter, plotdir,args.plot_name)
        except:
            print("Plotting loss failed")
        print("Epoch finished: %d -- Loss: %.6f -- Accuracy: %f" % (epoch, loss,accuracy))
        toc()   

        '''Save model checkpoint'''
        torch.save({
                    'epoch': epoch,
                    'loss_iter': loss_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, savedir + 'model_e%d' % (epoch) +args.plot_name+'.pt')
        
        '''Plot spin-wave propagation'''
        if model.retain_history:
            with torch.no_grad():
                try:
                    spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir,plotname=args.plot_name)
                    # mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
                    # wave_snapshot(model, mz[timesteps-1], (plotdir+'snapshot_time%d_epoch%d.png' % (timesteps,epoch)),r"$m_z$")
                    # wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+'snapshot_time%d_epoch%d.png' % (int(timesteps/2),epoch)),r"$m_z$")
                    # wave_integrated(model, mz, (plotdir+'integrated_epoch%d.png' % (epoch)))
                except:
                    print("Plotting failed")
if __name__ == '__main__':
    focus(parseArgs())