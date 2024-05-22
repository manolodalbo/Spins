import numpy as np
import tensorflow as tf
import torch
import pickle
from torch import tensor
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_preprocess_data():
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.mnist.load_data()
    train_inputs = train_inputs.reshape(-1,784)/255
    test_inputs = test_inputs.reshape(-1,784)/255
    train_inputs = fm(train_inputs,0.05e9,9e9)
    test_inputs = fm(test_inputs,0.05e9,9e9)
    train_labels = tensor(train_labels)
    print(train_labels.shape)
    print(train_inputs.shape)
    test_labels = tensor(test_labels)
    with open(f'C:/spin/data/data.p', 'wb') as pickle_file:
        pickle.dump(dict(train_inputs=train_inputs,train_labels=train_labels,test_inputs=test_inputs,test_labels=test_labels), pickle_file)
    print(f'Data has been dumped into {"C:/spin/data"}/data.p!')
def fm(inputs:np.array,Fi,Ff)-> np.array:
    dt = 20e-12     # timestep (s)
    timesteps = 600
    t = np.arange(0, timesteps * dt, dt)  # time vector
    modulation_index = 10
    carrier_freq = np.mean([Fi, Ff])  # Calculate carrier frequency for each batch
    modulating_signals = Fi + (Ff - Fi) * inputs
    modulated_wave = np.zeros((inputs.shape[0], len(t)))
    pbar = tqdm(inputs)
    for i in range(inputs.shape[0]):
        pbar.set_description(f"[({i+1}/{len(inputs)})] Processing images into waves")
        for j, modulating_signal in enumerate(modulating_signals[i]):
            modulated_wave[i] += np.sin(2 * np.pi * (carrier_freq + modulation_index * modulating_signal) * t)
    return modulated_wave
load_and_preprocess_data()