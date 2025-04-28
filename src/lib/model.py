import numpy as np

from src.lib.optimizers import SGD, Adam

class model:
    def __init__(self, layers):
        self.layers = layers
        self.layers[-1].model = self
        
    def forward(self, input, training = True):
        self.input = input
        loss = input
        
        for layer in self.layers:
            loss = layer.forward(loss, training = training)

        accuracy = self.layers[-1].accuracy
        
        self.loss = loss
        self.accuracy = accuracy
        
        return loss
    
    def backward(self):
        grad_output = self.loss
        
        for layer in self.layers[::-1]:
            grad_output = layer.backward(grad_output)
            
    def predict(self, input):
        output = input
        for layer in self.layers[:-1]:
            output = layer.forward(output)
            
        output = self.layers[-1].predict(output)
        return output
            
    def update_params(self, optimizer = SGD(learning_rate = 0.01)):
        for layer in self.layers:
            if hasattr(layer, "gradients"):
                gradients = layer.gradients
                params = layer.get_params()
                
                for param_id in params.keys():
                    new_param = optimizer(params[param_id], gradients[param_id], param_id)
                    params[param_id] = new_param
                    
                layer.set_params(params)
            
    def intialise_targets(self, targets):
        self.layers[-1].target = targets
        
        
    