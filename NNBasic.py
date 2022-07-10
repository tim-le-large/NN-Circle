from random import random

import numpy as np
import pygame
from math import exp

from PyGame import PyGame


class NNBasic:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.network = list()
        hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in
                        range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in
                        range(n_outputs)]
        self.network.append(output_layer)
        self.pygame = PyGame()

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i_weights in range(len(weights) - 1):
            activation += weights[i_weights] * inputs[i_weights]
        return activation

    # Transfer neuron activation
    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        # for every layer
        for i_layer in reversed(range(len(self.network))):
            layer = self.network[i_layer]
            # last layer
            if i_layer == len(self.network) - 1:
                # for every neuron in layer
                for i_neuron in range(len(layer)):
                    neuron = layer[i_neuron]
                    # error = (output - expected)
                    neuron["error"] = neuron['output'] - expected[i_neuron]
            # if layer ist not last
            else:
                # for every neuron on this layer
                for i_neuron in range(len(layer)):
                    error = 0.0
                    # for every neuron in [layer + 1]
                    for neuron in self.network[i_layer + 1]:
                        # multiply the neuron weights from layer[neuron] to layer+1[neuron] with the delta
                        error += (neuron['weights'][i_neuron] * neuron['delta'])
                    layer[i_neuron]["error"] = error
            # neuron[delta] = neuron[error] * o'(neuron[output])

            for i_neuron in range(len(layer)):
                neuron = layer[i_neuron]
                neuron['delta'] = neuron["error"] * self.sigmoid_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i_layer in range(len(self.network)):
            inputs = row[:-1]
            if i_layer != 0:
                inputs = [neuron['output'] for neuron in self.network[i_layer - 1]]
            for neuron in self.network[i_layer]:
                for i_inputs in range(len(inputs)):
                    neuron['weights'][i_inputs] -= l_rate * neuron['delta'] * inputs[i_inputs]
                neuron['weights'][-1] -= l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(self, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [row[-1]]
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
                # self.visualize_network(row)
            if epoch % 100 == 0:
                self.visualize_epoch()
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    def visualize_epoch(self):
        for x in np.arange(-1.0, 1.0, 0.1, dtype=float):
            for y in np.arange(-1.0, 1.0, 0.1, dtype=float):
                prediction = self.forward_propagate([x, y, 0.8])
                # print(prediction)
                self.pygame.draw(200 + x * 100, 200 + y * 100,
                                 (255 * prediction[0], 255 * prediction[0], 255 * prediction[0]))
