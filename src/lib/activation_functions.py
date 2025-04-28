import numpy as np 
from typing import List

class ReLu:
    def __init__(self):
        pass
    
    def forward(self, input : List[float], training=True):
        self.input = input
        self.output = np.maximum(0, input)
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output*(self.input > 0)
        
        return self.grad_input
    
class softmax:
    def __init__(self):
        pass 
    
    def forward(self, input : List[float], training=True):
        self.input = input
        maximum = np.max(input, axis=1, keepdims=True)
        exp_shifted = np.exp(input - maximum)
        normalisation_divisor = np.sum(exp_shifted, axis=1, keepdims=True)
        self.output = exp_shifted / normalisation_divisor
        
        return self.output
    
    def backward(self, grad_output):
        self.grad_input = grad_output
        return self.grad_input