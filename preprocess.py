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
    refined_inputs  = []
    refined_ouputs = []
    # already_seen = set()
    # i = 0
    # while len(refined_inputs) < 10:
    #     if train_labels[i] not in already_seen:
    #         refined_inputs.append(train_inputs[i])
    #         refined_ouputs.append(train_labels[i])
    #         already_seen.add(train_labels[i])
    #     i  = i + 1
    ones = 0
    zeros = 0
    i = 0
    while ones < 1 or zeros < 1:
        if train_labels[i] == 1 and ones < 1:
            refined_inputs.append(train_inputs[i])
            refined_ouputs.append(train_labels[i])
            ones = ones + 1
        if train_labels[i] == 0 and zeros < 1:
            refined_inputs.append(train_inputs[i])
            refined_ouputs.append(train_labels[i])
            zeros = zeros + 1
        i += 1
        
    train_inputs = fm(np.array(refined_inputs),0.2e9,5e9,10)
    test_inputs = fm(test_inputs[0:10],0.2e9,5e9,10)
    train_labels = tensor(refined_ouputs)
    print(train_labels.shape)
    print(train_inputs.shape)
    test_labels = tensor(test_labels)
    with open(f'C:/spins/data/data.p', 'wb') as pickle_file:
        pickle.dump(dict(train_inputs=train_inputs,train_labels=train_labels,test_inputs=test_inputs,test_labels=test_labels[0:10]), pickle_file)
    print(f'Data has been dumped into {"C:/spins/data"}/data.p!')
def fm(inputs: np.array, Fi: float, Ff: float,samples_per_point:int) -> np.array:
    """
    Frequency modulate the input images.
    
    Parameters:
        inputs (np.array): Array of shape (number of inputs, 784), where each row represents an image.
        Fi (float): Minimum frequency in Hz.
        Ff (float): Final frequency in Hz.
    
    Returns:
        np.array: Frequency modulated waveforms for each input image.
    """
    points_per_input = samples_per_point
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
                modulated_wave[i, points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (1))*np.sin(2 * np.pi * frequency * t[1:points_per_input+1] + phase)
                if np.cos(2*np.pi*frequency*t[points_per_input] + phase) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency * t[points_per_input] + phase)
            else:
                modulated_wave[i,points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (1))*np.sin(2*np.pi*frequency*t[0:points_per_input])
                if np.cos(2*np.pi*frequency*t[points_per_input-1]) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency *t[points_per_input-1])
        plt.figure()
        plt.plot(modulated_wave[i])
        plt.show()
    return modulated_wave
load_and_preprocess_data()