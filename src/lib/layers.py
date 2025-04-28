import numpy as np 
from typing import List

from src.lib.initializers import HeInitializer, XavierInitializer, GlorotUniformInitializer, ZeroInitializer, OneInitializer
from src.lib.optimizers import SGD, Adam

class Dense:
    def __init__(self, input_size : int, output_size : int, weights : List[List[float]] = None, biases : List[float] = None, initializer = None, float_type = np.float32, eps = 1e-15):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weights
        self.biases = biases
        self.eps = eps
        self.gradients = {}
        
        weights_id = f'{id(self)}_weights'
        biases_id = f'{id(self)}_biases'
        ids = {"weights" : weights_id,
               "biases" : biases_id}
        
        self.param_ids = ids
        
        self.initialise_weights(initializer=initializer, float_type = float_type)
        
    def initialise_weights(self, initializer = None, float_type = np.float32):
        zero_initializer  = ZeroInitializer(dtype = float_type)
        if initializer is None:
            initializer = GlorotUniformInitializer(dtype = float_type)
            
        self.weights = initializer((self.input_size, self.output_size))
        self.biases = zero_initializer(self.output_size)
    
    def forward(self, input, training=True):
        self.input = input
        self.output = input @ self.weights + self.biases
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output @ self.weights.T
        
        self.gradients[self.param_ids["weights"]] = self.input.T @ grad_output
        self.gradients[self.param_ids["biases"]] = self.grad_biases = np.sum(grad_output, axis=0)
        
        return self.grad_input
    
    def get_params(self):
        params = {}
        
        params[self.param_ids["weights"]] = self.weights
        params[self.param_ids["biases"]] = self.biases
        
        return params
    
    def set_params(self, params):        
        self.weights = params[self.param_ids["weights"]]
        self.biases = params[self.param_ids["biases"]]

        
class flatten:
    def __init__(self):
        pass
        
    def forward(self, input, training=True):
        self.input = input
        self.input_shape = input.shape
        self.output = input.reshape(self.input_shape[0], -1)
        
        return self.output
        
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
        

class BatchNorm1D:
    def __init__(self, input_size, gamma = None, beta = None, momentum = 0.99, eps = 1e-15, float_type = np.float32):
        self.gamma = gamma
        self.beta = beta
        self.input_size = input_size
        self.momentum = momentum
        self.eps = eps
        self.gradients = {}
        
        weights_id = f'{id(self)}_gamma'
        biases_id = f'{id(self)}_beta'
        ids = {"gamma" : weights_id,
               "beta" : biases_id}
        
        self.param_ids = ids
        
        self.initialize_weights(float_type=float_type)
        
    def forward(self, input, training=True):
        self.input = input
        if training:
            batch_mean = np.mean(input, axis = 0)
            batch_std = np.std(input, axis = 0)
            normalized_input = (input - batch_mean) / (batch_std + self.eps)
            
            self.batch_std = batch_std
            self.batch_mean = batch_mean
            self.running_mean = (self.momentum*self.running_mean) + (1-self.momentum)*batch_mean
            self.running_std = (self.momentum*self.running_std) + (1-self.momentum)*batch_std
        else:
            normalized_input = (input - self.running_mean) / (self.running_std + self.eps)
        
        self.normalized_input = normalized_input
        output = self.gamma*normalized_input + self.beta
        
        return output
    
    def backward(self, grad_output):
        self.gradients[self.param_ids["beta"]] = np.sum(grad_output, axis = 0)
        self.gradients[self.param_ids["gamma"]] = np.sum(grad_output * self.normalized_input, axis=0)
        
        N, D = grad_output.shape

        grad_x_hat = grad_output * self.gamma

        grad_mean = np.sum(grad_x_hat, axis=0)
        grad_var = np.sum(grad_x_hat * self.normalized_input, axis=0)

        self.grad_input = (1. / N) * (1. / (self.batch_std + self.eps)) * (
            N * grad_x_hat
            - grad_mean
            - self.normalized_input * grad_var
        )
        
        return self.grad_input
        
    def initialize_weights(self, float_type = np.float32):
        zero_initializer = ZeroInitializer(dtype = float_type)
        one_initializer = OneInitializer(dtype = float_type)
        
        self.gamma = one_initializer(self.input_size)
        self.beta = zero_initializer(self.input_size)
        self.running_mean = zero_initializer(self.input_size)
        self.running_std = one_initializer(self.input_size)
        
    def get_params(self):
        params = {}
        
        params[self.param_ids["gamma"]] = self.gamma
        params[self.param_ids["beta"]] = self.beta
        
        return params
    
    def set_params(self, params):        
        self.gamma = params[self.param_ids["gamma"]]
        self.beta = params[self.param_ids["beta"]]
