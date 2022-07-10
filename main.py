# import the necessary packages

# construct the XOR dataset

from Data import Data
from NNBasic import NNBasic

if __name__ == '__main__':
    dataset = Data().generate_circle_data(4000)
    n_outputs = 1
    n_inputs = len(dataset[0]) - n_outputs
    nn = NNBasic(n_inputs, 4, n_outputs)
    nn.train_network(dataset, 0.1, 1000, n_outputs)
