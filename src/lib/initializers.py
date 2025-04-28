import cupy as np

class Initializer:
    def __init__(self, dtype=np.float64):
        self.quantizer = Quantizer(dtype)

    def __call__(self, shape):
        raise NotImplementedError("Initializer must implement the __call__ method.")

class HeInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(np.random.randn(*shape) * np.sqrt(2. / shape[0]))

class XavierInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(np.random.randn(*shape) * np.sqrt(2. / (shape[0] + shape[1])))

class GlorotUniformInitializer(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return self.quantizer(np.random.uniform(-limit, limit, size=shape))

class ZeroInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(np.zeros(shape))

class OneInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(np.ones(shape))

class Quantizer:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        
    def __call__(self, input):
        return input.astype(self.dtype)
