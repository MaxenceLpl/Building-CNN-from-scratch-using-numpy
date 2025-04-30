from src.lib.backend import backend, HasBackend
from typing import List

class ActivationFunction(HasBackend):
    pass

class ReLu(ActivationFunction):
    def __init__(self):
        pass
    
    def forward(self, input : List[float], training=True):
        self.input = input
        self.output = self.xp.maximum(0, input)
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output*(self.input > 0)
        
        return self.grad_input

    def get_output_shape(self, input_shape):
        return input_shape
    
class softmax(ActivationFunction):
    def __init__(self):
        pass 
    
    def forward(self, input : List[float], training=True):
        self.input = input
        maximum = self.xp.max(input, axis=1, keepdims=True)
        exp_shifted = self.xp.exp(input - maximum)
        normalisation_divisor = self.xp.sum(exp_shifted, axis=1, keepdims=True)
        self.output = exp_shifted / normalisation_divisor
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output
        return self.grad_input
    
    def get_output_shape(self, input_shape):
        return input_shape