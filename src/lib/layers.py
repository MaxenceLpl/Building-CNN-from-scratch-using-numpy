import numpy as np 
from typing import List

from src.lib.initializers import HeInitializer, XavierInitializer, GlorotUniformInitializer, ZeroInitializer, OneInitializer
from src.lib.optimizers import SGD, Adam

class Dense:
    def __init__(self, output_size : int, initializer = None, eps = 1e-15):
        self.output_size = output_size
        self.eps = eps
        self.gradients = {}
        self.initializer = initializer
        
        weights_id = f'{id(self)}_weights'
        biases_id = f'{id(self)}_biases'
        ids = {"weights" : weights_id,
               "biases" : biases_id}
        
        self.param_ids = ids
        
    def initialize_weights(self, float_type = np.float64):
        zero_initializer  = ZeroInitializer(dtype = float_type)
        if self.initializer is None:
            self.initializer = GlorotUniformInitializer(dtype = float_type)
            
        self.weights = self.initializer((self.input_size, self.output_size))
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
        
    def get_output_shape(self, input_shape):
        self.input_size = input_shape[0]
        return (self.output_size,)

        
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
        
    def get_output_shape(self, input_shape):
        return (np.prod(input_shape),)

class BatchNorm1D:
    def __init__(self, momentum = 0.99, eps = 1e-15):
        self.momentum = momentum
        self.eps = eps
        self.gradients = {}
        
        weights_id = f'{id(self)}_gamma'
        biases_id = f'{id(self)}_beta'
        ids = {"gamma" : weights_id,
               "beta" : biases_id}
        
        self.param_ids = ids
        
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
        
    def initialize_weights(self, float_type = np.float64):
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
        
    def get_output_shape(self, input_shape):
        self.input_size = input_shape[0]
        return input_shape

class Conv2D:
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, initializer=None):
        self.input_shape = None
        self.out_channels = out_channels
        self.initializer = initializer

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.gradients = {}
        self.indices_built = False  # NEW: pour savoir si on a précalculé les indices

        weights_id = f'{id(self)}_weights'
        biases_id = f'{id(self)}_biases'
        self.param_ids = {"weights": weights_id, "biases": biases_id}

    def initialize_weights(self, float_type=np.float64):
        zero_initializer = ZeroInitializer(dtype=float_type)
        if self.initializer is None:
            self.initializer = GlorotUniformInitializer(dtype=float_type)

        self.weights = self.initializer((self.out_channels, self.in_channels) + self.kernel_size)
        self.biases = zero_initializer(self.out_channels)

        self.weights = self.weights.reshape(self.out_channels, -1)
        self.biases = self.biases.reshape(1, -1)
        
        self.in_channels = self.input_shape[0]

    def forward(self, input, training=True):
        if len(input.shape) == 3:
            input = input[:, np.newaxis, :, :]  # Ajoute channel si besoin

        self.input = input
        N, C, H, W = input.shape

        if not self.indices_built:
            self._build_indices(C, H, W)

        x_cols = self._im2col_fast(input)

        self.x_cols = x_cols  # on garde pour le backward

        out = x_cols @ self.weights.T + self.biases
        out = out.reshape(N, self.height_out, self.width_out, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_output):
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.gradients[self.param_ids["biases"]] = np.sum(grad_output_reshaped, axis=0)
        self.gradients[self.param_ids["weights"]] = grad_output_reshaped.T @ self.x_cols

        grad_input_cols = grad_output_reshaped @ self.weights
        grad_input = self._col2im_fast(grad_input_cols)
        return grad_input

    def get_params(self):
        return {self.param_ids["weights"]: self.weights, self.param_ids["biases"]: self.biases}

    def set_params(self, params):
        self.weights = params[self.param_ids["weights"]]
        self.biases = params[self.param_ids["biases"]]

    def _build_indices(self, C, H, W):
        KH, KW = self.kernel_size
        stride, padding = self.stride, self.padding

        i0 = np.repeat(np.arange(KH), KW)
        i0 = np.tile(i0, C)
        j0 = np.tile(np.arange(KW), KH * C)

        i1 = stride * np.repeat(np.arange(self.height_out), self.width_out)
        j1 = stride * np.tile(np.arange(self.width_out), self.height_out)

        self.i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        self.j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        self.k = np.repeat(np.arange(C), KH * KW).reshape(-1, 1)

        self.indices_built = True

    def _im2col_fast(self, input):
        N, C, H, W = input.shape
        KH, KW = self.kernel_size
        stride, padding = self.stride, self.padding

        # Padding
        input_padded = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)))

        H_padded, W_padded = input_padded.shape[2], input_padded.shape[3]

        # Strides setup
        shape = (N, C, KH, KW, self.height_out, self.width_out)
        strides = (
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2],
            input_padded.strides[3],
            input_padded.strides[2] * stride,
            input_padded.strides[3] * stride
        )

        patches = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides, writeable=False)

        # Reshape pour avoir (N * H_out * W_out, C * KH * KW)
        patches_reshaped = patches.transpose(0, 4, 5, 1, 2, 3).reshape(N * self.height_out * self.width_out, -1)

        return patches_reshaped

    def _col2im_fast(self, cols):
        N, C, H, W = self.input.shape
        KH, KW = self.kernel_size
        stride, padding = self.stride, self.padding
        H_out, W_out = self.height_out, self.width_out

        x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

        # Remise en forme : (N, H_out, W_out, C, KH, KW)
        cols_reshaped = cols.reshape(N, H_out, W_out, C, KH, KW)

        for y in range(KH):
            for x in range(KW):
                x_padded[:, :, 
                        y:y + stride * H_out:stride, 
                        x:x + stride * W_out:stride] += cols_reshaped[:, :, :, :, y, x].transpose(0, 3, 1, 2)

        if padding == 0:
            return x_padded[:, :, :H, :W]
        else:
            return x_padded[:, :, padding:-padding, padding:-padding]

        
    def get_output_shape(self, input_shape):
        channels_in, height_in, width_in = input_shape
        KH, KW = self.kernel_size
        stride, padding = self.stride, self.padding

        self.height_out = (height_in + 2 * padding - KH) // stride + 1
        self.width_out = (width_in + 2 * padding - KW) // stride + 1
        self.in_channels = input_shape[0]

        return (self.out_channels, self.height_out, self.width_out)


class MaxPooling2D:
    def __init__(self, kernel_size, stride=1):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training = True):
        self.input = input
        N, C, H, W = input.shape
        KH, KW = self.kernel_size
        S = self.stride

        H_out = (H - KH) // S + 1
        W_out = (W - KW) // S + 1

        # Découpe en patches
        x_reshaped = np.lib.stride_tricks.as_strided(
            input,
            shape=(N, C, H_out, W_out, KH, KW),
            strides=(
                input.strides[0],
                input.strides[1],
                S * input.strides[2],
                S * input.strides[3],
                input.strides[2],
                input.strides[3]
            ),
            writeable=False
        )

        # On reshape pour vectoriser
        self.x_patches = x_reshaped.reshape(N, C, H_out, W_out, -1)

        # Store les indices des max
        self.max_indices = self.x_patches.argmax(axis=-1)

        # Prendre max
        out = self.x_patches.max(axis=-1)
        return out

    def backward(self, grad_output):
        N, C, H_out, W_out = grad_output.shape
        KH, KW = self.kernel_size
        S = self.stride
        H, W = self.input.shape[2], self.input.shape[3]

        # Initialiser le gradient d'entrée à zéro
        grad_input = np.zeros_like(self.input)

        # Indices pour placer correctement grad_output
        grad_output_flat = grad_output.flatten()

        # Calculer les indices où propager
        # On construit les matrices d'indices globaux pour chaque dimension
        n_idx = np.repeat(np.arange(N), C * H_out * W_out)
        c_idx = np.tile(np.repeat(np.arange(C), H_out * W_out), N)
        h_idx = np.tile(np.repeat(np.arange(H_out), W_out), N * C) * S
        w_idx = np.tile(np.tile(np.arange(W_out), H_out), N * C) * S

        # Offset dans le patch (max position)
        offset = self.max_indices.flatten()

        # Où dans le patch (KH x KW) on doit aller
        offset_h = offset // KW
        offset_w = offset % KW

        # Positions finales dans l'image d'entrée
        final_h = h_idx + offset_h
        final_w = w_idx + offset_w

        # Ajouter les gradients aux bonnes positions
        np.add.at(grad_input, (n_idx, c_idx, final_h, final_w), grad_output_flat)

        return grad_input

    def get_output_shape(self, input_shape):
        channels, height_in, width_in = input_shape
        KH, KW = self.kernel_size
        stride = self.stride

        height_out = (height_in - KH) // stride + 1
        width_out = (width_in - KW) // stride + 1

        return (channels, height_out, width_out)