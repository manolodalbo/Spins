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
        #it might be worthwhile to crop the output to only once both inputs have gone
        # it might also make sense to just weight later outputs more strongly. linearly increasing weights?
        film_output = film_output.sum(dim=-1) #need to do some sort of normalization here to get it in the general range [-5,5]. Need to do analysis on what typical std and mean are to normalize
        distances = self.euclidean_distance(film_output) #batch_size x vocab_size
        probs = 1 - distances/distances.sum(dim=-1).unsqueeze(0)
        return probs
    def euclidean_distance(self,outputs):
        """"
        Note: not actually euclidean distance because I don't take the square root after adding
        Should takes in the the output after passing through film and normalized(shape batch_size x embed_size) and compares
        with the embedding matrix. Returns batch_size x vocab_size
        """
        distances = outputs.unsqueeze(1) - self.embedding_matrix.unsqueeze(0) #bs X vocab_size X embed_size
        distances = (distances ** 2).sum(dim = -1) #essentially gets mean squared error(this may not be necessary)

        #there are two ways to do this:
        # the first is to use the distances to compute logits and the second is to directly
        # use the mean squared error to compute the loss, however, the embeddings would have
        # to be fixed because or else all the embeddings could be the same and would still result in 0 loss every time
        # i dont think MSE will work in this case without computing probs later
        # useful bc it gets rid of negatives and punishes terrible guesses
        return distances

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
    print(Y0)
    print(Y0.shape)
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