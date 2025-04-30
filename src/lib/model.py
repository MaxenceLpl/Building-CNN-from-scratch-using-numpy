from src.lib.backend import backend, HasBackend
from src.lib.optimizers import SGD, Adam

class Model(HasBackend):
    pass

class model(Model):
    def __init__(self, layers, input_shape, float_type = None):
        self.layers = layers
        self.layers[-1].model = self
        
        self.input_shape = input_shape
        float_type = float_type if float_type is not None else self.xp.float32
        
        self.initialize_input_shapes()
        self.initialize_weights(float_type=float_type)
        
    def initialize_input_shapes(self):
        input_shape = self.input_shape[1:]
        
        for layer in self.layers:
            layer.input_shape = input_shape
            input_shape = layer.get_output_shape(input_shape)
            
    def initialize_weights(self, float_type = None):
        float_type = float_type if float_type is not None else self.xp.float32
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
        
    def summary(self):
        print("┌" + "─" * 70 + "┐")
        print(f"{'Layer (type)':<20} {'Input Shape':<25} {'Output Shape':<25}")
        print("├" + "─" * 20 + "┬" + "─" * 25 + "┬" + "─" * 25 + "┤")

        for layer in self.layers:
            name = layer.__class__.__name__
            in_shape = str(layer.input_shape)
            out_shape = str(layer.get_output_shape(layer.input_shape))
            print(f"{name:<20} {in_shape:<25} {out_shape:<25}")

        print("└" + "─" * 70 + "┘")

        
        
    