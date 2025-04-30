from src.lib.backend import backend, HasBackend

class Initializer(HasBackend):
    def __init__(self, dtype=None):
        dtype = dtype if dtype is not None else self.xp.float32
        self.quantizer = Quantizer(dtype)

    def __call__(self, shape):
        raise NotImplementedError("Initializer must implement the __call__ method.")

class HeInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(self.xp.random.randn(*shape) * self.xp.sqrt(2. / shape[0]))

class XavierInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(self.xp.random.randn(*shape) * self.xp.sqrt(2. / (shape[0] + shape[1])))

class GlorotUniformInitializer(Initializer):
    def __call__(self, shape):
        limit = self.xp.sqrt(6 / (shape[0] + shape[1]))
        return self.quantizer(self.xp.random.uniform(-limit, limit, size=shape))

class ZeroInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(self.xp.zeros(shape))

class OneInitializer(Initializer):
    def __call__(self, shape):
        return self.quantizer(self.xp.ones(shape))

class Quantizer(HasBackend):
    def __init__(self, dtype=None):
        dtype = dtype if dtype is not None else self.xp.float32
        self.dtype = dtype
        
    def __call__(self, input):
        return input.astype(self.dtype)
