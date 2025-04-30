from src.lib.backend import backend, HasBackend
from typing import List

class LossFunction(HasBackend):
    pass

class CrossEntropy(LossFunction):
    def __init__(self, target=None, l2_lambda = 1e-8, model = None):
        self.target = target
        self.l2_lambda = l2_lambda
        self.model = model
        
    def forward(self, prediction, training=True):
        self.logits = prediction
        
        # Trick pour stabilité numérique
        logits_shifted = prediction - self.xp.max(prediction, axis=1, keepdims=True)
        log_sum_exp = self.xp.log(self.xp.sum(self.xp.exp(logits_shifted), axis=1, keepdims=True))
        
        # Formule fusionnée softmax+crossentropy
        cross_entropy = -self.xp.sum(self.target * (logits_shifted - log_sum_exp), axis=1)
        self.loss = self.xp.mean(cross_entropy)

        # Ajouter L2
        l2_loss = 0
        if self.l2_lambda > 0 and self.model is not None:
            for layer in self.model.layers:
                if hasattr(layer, "weights"):
                    l2_loss += self.xp.sum(layer.weights ** 2)
            self.loss += (self.l2_lambda / 2) * l2_loss

        self.calculate_accuracy()
        return self.loss
    
    def predict(self, prediction):
        exps = self.xp.exp(prediction - self.xp.max(prediction, axis=1, keepdims=True))
        self.prediction = exps / self.xp.sum(exps, axis=1, keepdims=True)

        # Compute cross-entropy loss
        eps = 1e-15
        self.prediction = self.xp.clip(self.prediction, eps, 1 - eps)
        return self.prediction
    
    def backward(self, loss=0):
        batch_size = self.logits.shape[0]

        # Refaire softmax uniquement ici
        exps = self.xp.exp(self.logits - self.xp.max(self.logits, axis=1, keepdims=True))
        softmax = exps / self.xp.sum(exps, axis=1, keepdims=True)

        self.grad_input = (softmax - self.target) / batch_size
        return self.grad_input
    
    def calculate_accuracy(self):
        exps = self.xp.exp(self.logits - self.xp.max(self.logits, axis=1, keepdims=True))
        softmax = exps / self.xp.sum(exps, axis=1, keepdims=True)

        pred_labels = self.xp.argmax(softmax, axis=1)
        true_labels = self.xp.argmax(self.target, axis=1)

        acc = self.xp.mean(pred_labels == true_labels)
        self.accuracy = acc
        return acc

    def get_output_shape(self, input_shape):
        self.nb_of_classes = input_shape[0]
        return self.nb_of_classes
