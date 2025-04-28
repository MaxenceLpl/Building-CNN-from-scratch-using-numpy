import numpy as np

from src.lib.optimizers import SGD, Adam

class model:
    def __init__(self, layers, input_shape, float_type = np.float32):
        self.layers = layers
        self.layers[-1].model = self
        
        self.input_shape = input_shape
        
        self.initialize_input_shapes()
        self.initialize_weights(float_type=float_type)
        
    def initialize_input_shapes(self):
        input_shape = self.input_shape[1:]
        
        for layer in self.layers:
            print(input_shape)
            layer.input_shape = input_shape
            input_shape = layer.get_output_shape(input_shape)
            
    def initialize_weights(self, float_type = np.float64):
        for layer in self.layers:
            if hasattr(layer, "initialize_weights"):
                layer.initialize_weights(float_type = float_type)
        
    def forward(self, input, training = True):
        self.input = input
        loss = input
        
        import time
        
        for layer in self.layers:
            start = time.time()
            loss = layer.forward(loss, training = training)
            #print(f"{layer.__class__.__name__} : {time.time() - start}")

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
        
        
    