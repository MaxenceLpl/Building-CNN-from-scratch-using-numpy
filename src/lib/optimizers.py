from src.lib.backend import backend, HasBackend

class Optimizer(HasBackend):
    def __call__(self, param, grad, key):
        raise NotImplementedError("Optimizer must implement the call method.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def __call__(self, param, grad, key=None):
        return param - self.learning_rate * grad

class AdamGPU(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}   # buffers 1er moment
        self.v = {}   # buffers 2e moment
        self.t = {}   # timestep par paramètre
        
        self.set_kernel()

    def __call__(self, param, grad, key):
        # Initialisation au premier appel
        if key not in self.m:
            self.m[key] = self.xp.zeros_like(param)
            self.v[key] = self.xp.zeros_like(param)
            self.t[key] = 0

        # On incrémente le compteur
        self.t[key] += 1

        # Appel du kernel unique qui fait tout en GPU
        param_updated, m_new, v_new = self._adam_kernel(
            param, grad,
            self.learning_rate,
            self.beta1, self.beta2, self.eps,
            self.t[key],
            self.m[key], self.v[key]
        )

        # On stocke les nouveaux moments
        self.m[key] = m_new
        self.v[key] = v_new

        return param_updated
    
    def set_kernel(self):
        self._adam_kernel = self.xp.ElementwiseKernel(
        # inputs :
        'T param, T grad, T lr, T beta1, T beta2, T eps, int32 t, T m, T v',
        # outputs (trois sorties) :
        'T param_out, T m_out, T v_out',
        '''
        // calcul des moments corrigés
        T m_new = beta1 * m + (1 - beta1) * grad;
        T v_new = beta2 * v + (1 - beta2) * grad * grad;
        T m_hat = m_new / (1 - pow(beta1, t));
        T v_hat = v_new / (1 - pow(beta2, t));
        // mise à jour du paramètre
        param_out = param - lr * m_hat / (sqrt(v_hat) + eps);
        // on renvoie aussi les nouveaux moments
        m_out = m_new;
        v_out = v_new;
        ''',
        'adam_update_fused'
    )
    
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}   # 1er moment
        self.v = {}   # 2e moment
        self.t = {}   # compteur de pas

    def __call__(self, param, grad, key):
        # Initialisation au premier appel pour ce paramètre
        if key not in self.m:
            self.m[key] = self.xp.zeros_like(param)
            self.v[key] = self.xp.zeros_like(param)
            self.t[key] = 0

        # Incrémenter le pas
        self.t[key] += 1
        t = self.t[key]

        # Mise à jour des moments
        m = self.m[key]
        v = self.v[key]
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad * grad

        # Correction de biais
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        # Mise à jour du paramètre
        param_out = param - self.learning_rate * m_hat / (self.xp.sqrt(v_hat) + self.eps)

        # Sauvegarder les nouveaux moments
        self.m[key] = m
        self.v[key] = v

        return param_out
