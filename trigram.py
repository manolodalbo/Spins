from preprocess import get_data,turn_into_wave
import numpy as np
import torch
import torch.nn as nn
from SpinTorch.spintorch.RNN_film import RNN_film
class MyTrigram(nn.Module):
    def __init__(self,vocab_size,batch_size,embed_size=80):
        super(MyTrigram,self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size

        self.film_RNN = RNN_film(self.embed_size,self.batch_size)
        self.embedding_matrix = nn.Parameter(torch.normal(torch.zeros(self.vocab_size,self.embed_size),std=5))
    def forward(self,inputs):
        wave_inputs = turn_into_wave(inputs,self.embedding_matrix)
        film_output = self.film_RNN(wave_inputs)
        print(film_output.shape)
        print("called")
def perplexity(labels,preds):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    calculate_categorical = loss_fn(preds, labels)
    
    # Divide by batch size (calculate the mean)
    divide_by_batch_size = calculate_categorical.mean()
    
    # Return the exponential of the mean loss
    return torch.exp(divide_by_batch_size)

def main():
    epochs = 10
    batch_size = 64
    embed_size = 80
    learning_rate = 0.01

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)  # 'cuda' or 'cpu'

    data_path = './data'
    train_tokens,test_tokens,vocab = get_data(f"{data_path}/train.txt",f"{data_path}/test.txt")
    train_array = np.array(train_tokens)
    test_array = np.array(test_tokens)
    X0, Y0  = np.vstack([train_array[0:-2],train_array[1:-1]]).T, train_array[2:]
    X1, Y1  = np.vstack([test_array[0:-2],test_array[1:-1]]).T, test_array[2:]
    model = MyTrigram(len(vocab),batch_size,embed_size=embed_size).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for i in range(0,len(X0),batch_size):
            inputs = torch.tensor(X0[i:i + batch_size], dtype=torch.float32).to(dev)
            targets = torch.tensor(Y0[i:i + batch_size], dtype=torch.long).to(dev)
            optimizer.zero_grad()
            outputs = model(inputs) 
            blablal
            loss = criterion()
            loss.backward()
            loss.step()         

if __name__ == '__main__':
    main()