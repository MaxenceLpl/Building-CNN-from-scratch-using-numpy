import numpy as np 
from typing import List

class Dense:
    def __init__(self, input_size : int, output_size : int, weights : List[List[float]] = None, biases : List[float] = None, initialisation : str = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weights
        self.biases = biases
        
        self.initialise_weights(initialisation=initialisation)
        
    def initialise_weights(self, initialisation : str = 'he'):
        if initialisation == 'he':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2. / self.input_size)
        elif initialisation == 'xavier':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2. / (self.input_size + self.output_size))
        elif initialisation == 'glorot':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = np.random.uniform(-limit, limit, size=(self.input_size, self.output_size))
            
        self.biases = np.zeros(self.output_size)
    
    def forward(self, input):
        self.input = input
        self.output = input @ self.weights + self.biases
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output @ self.weights.T
        self.grad_weights = self.input.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0)
        
        return self.grad_input
    
    def update_params(self, learning_rate):
        self.weights -= learning_rate*self.grad_weights
        self.biases -= learning_rate*self.grad_biases
        
        
class flatten:
    def __init__(self):
        pass
        
    def forward(self, input):
        self.input = input
        self.input_shape = input.shape
        self.output = input.reshape(self.input_shape[0], -1)
        
        return self.output
        
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
        