import numpy as np

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, input):
        self.input = input
        loss = input
        
        for layer in self.layers:
            loss = layer.forward(loss)
            
        prediction = self.layers[-1].prediction
        accuracy = self.layers[-1].accuracy
        
        self.loss = loss
        self.prediction = prediction
        self.accuracy = accuracy
        
        return prediction, loss
    
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
            
    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                layer.update_params(learning_rate)
            
    def intialise_targets(self, targets):
        self.layers[-1].target = targets
        
        
    