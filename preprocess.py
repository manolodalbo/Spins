
import numpy as np
import tensorflow as tf
import torch
import pickle
from torch import tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
def parseArgs()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--points',type=int, default=3)
    parser.add_argument('--pooling',type=bool,default=False)
    parser.add_argument('--min_freq',type=float,default=0.1e9)
    parser.add_argument('--max_freq',type=float,default = 5e9)
    parser.add_argument('--size',type=int,default = 320)
    parser.add_argument('--all_classes',type=bool,default=False)
    args = parser.parse_args()
    return args
def load_and_preprocess_data(args:argparse.Namespace):
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.mnist.load_data()
    print(train_inputs.shape)
    if args.pooling:
        train_inputs = pool(train_inputs)
        test_inputs = pool(test_inputs)
    train_inputs = train_inputs.reshape(-1,train_inputs.shape[-1]*train_inputs.shape[-2])/255
    test_inputs = test_inputs.reshape(-1,test_inputs.shape[-1]*test_inputs.shape[-2])/255
    refined_inputs  = []
    refined_ouputs = []
    if args.all_classes == False:
        ones = 0
        zeros = 0
        i = 0
        while ones < 400 or zeros < 400:
            if train_labels[i] == 1 and ones < 400:
                refined_inputs.append(train_inputs[i])
                refined_ouputs.append(train_labels[i])
                ones = ones + 1
            if train_labels[i] == 0 and zeros < 400:
                refined_inputs.append(train_inputs[i])
                refined_ouputs.append(train_labels[i])
                zeros = zeros + 1
            i += 1
    else:
        refined_inputs = train_inputs[0:args.size]
        refined_ouputs = train_labels[0:args.size]
    new_test_inputs = []
    new_test_labels = []
    if args.all_classes == False:
        i = 0
        ones=0
        zeros=0
        number_of_samples_of_each_class = int(0.2 * args.size)//2 if int(0.2*args.size)>= 320 else 160
        while ones<number_of_samples_of_each_class or zeros<number_of_samples_of_each_class:
            if test_labels[i] == 1 and ones < number_of_samples_of_each_class:
                new_test_inputs.append(test_inputs[i])
                new_test_labels .append(test_labels[i])
                ones = ones + 1
            if test_labels[i] == 0 and zeros < number_of_samples_of_each_class:
                new_test_inputs.append(test_inputs[i])
                new_test_labels.append(test_labels[i])
                zeros = zeros + 1
            i+=1
    else:
        testing_size = int(0.2 * args.size) if int(0.2*args.size)>= 320 else 320
        new_test_inputs = test_inputs[0:testing_size]
        new_test_labels = test_labels[0:testing_size]
    train_inputs = fm(np.array(refined_inputs),args.min_freq,args.max_freq,args.points)
    test_inputs = fm(np.array(new_test_inputs),args.min_freq,args.max_freq,args.points)
    train_labels = tensor(refined_ouputs)
    test_labels = tensor(new_test_labels)
    print("train_inputs shape: " + train_inputs.shape)
    print("test_inputs shape: "  + test_inputs.shape)
    print("train_labels shape: "  + train_labels.shape)
    with open(f'C:/spins/data/data.p', 'wb') as pickle_file:
        pickle.dump(dict(train_inputs=train_inputs,train_labels=train_labels,test_inputs=test_inputs,test_labels=test_labels), pickle_file)
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
                modulated_wave[i, points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (2))*np.sin(2 * np.pi * frequency * t[1:points_per_input+1] + phase)
                if np.cos(2*np.pi*frequency*t[points_per_input] + phase) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency * t[points_per_input] + phase)
            else:
                modulated_wave[i,points_per_input*j:points_per_input*(j+1)] = (0.5 + pixel_intensity * (2))*np.sin(2*np.pi*frequency*t[0:points_per_input])
                if np.cos(2*np.pi*frequency*t[points_per_input-1]) > 0:
                    pos_deriv=True
                else:
                    pos_deriv = False
                prev = np.sin(2* np.pi * frequency *t[points_per_input-1])
    return modulated_wave
def pool(inputs:np.array):
    """Performs average pooling on image, effectively cutting down the resolution. For
    mnist this means going from 28 by 28 to 14 by 14"""
    inputs = tensor(inputs,dtype=torch.float32)
    show_image(inputs[2])
    inputs = inputs.unsqueeze(1)
    #the output width and heigh is governed by the following equation assuming no padding:
    #w_f = (w_i + filter_width)/stride and the same for heigh
    pooling_layer = torch.nn.AvgPool2d(kernel_size=(2,2),stride=2,padding=0)
    pooled = pooling_layer(inputs).squeeze()
    to_return = pooled.numpy()
    show_image(to_return[2])
    return to_return
def show_image(image:np.array):
    """
    Used to show mnist image """
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    load_and_preprocess_data(parseArgs())