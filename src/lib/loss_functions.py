import numpy as np 

from typing import List

import numpy as np

class CrossEntropy:
    def __init__(self, nb_of_classes, target=None):
        self.nb_of_classes = nb_of_classes
        self.target = target
        
    def forward(self, prediction):
        # Apply softmax
        exps = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        self.prediction = exps / np.sum(exps, axis=1, keepdims=True)

        # Compute cross-entropy loss
        eps = 1e-15
        self.prediction = np.clip(self.prediction, eps, 1 - eps)
        self.loss = -np.sum(self.target * np.log(self.prediction)) / self.prediction.shape[0]
        
        self.calculate_accuracy()
        return self.loss
    
    def predict(self, prediction):
        exps = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        self.prediction = exps / np.sum(exps, axis=1, keepdims=True)

        # Compute cross-entropy loss
        eps = 1e-15
        self.prediction = np.clip(self.prediction, eps, 1 - eps)
        return self.prediction
    
    def backward(self, loss=0):
        # Derivative of cross-entropy combined with softmax
        batch_size = self.prediction.shape[0]
        self.grad_input = (self.prediction - self.target) / batch_size
        return self.grad_input
    
    def calculate_accuracy(self):
        pred_labels = np.argmax(self.prediction, axis=1)
        true_labels = np.argmax(self.target, axis=1)
        acc = np.mean(pred_labels == true_labels)
        
        self.accuracy = acc
        return acc