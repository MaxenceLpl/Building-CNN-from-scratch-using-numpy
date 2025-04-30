# src/lib/backend.py
import os

# Si vous préférez contrôler via variable d'environnement :
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

# Import des deux bibliothèques
import numpy as _np
import cupy as _cp

class Backend:
    def __init__(self):
        # flag initial; peut être modifié à chaud avec set_gpu()
        self.use_gpu = USE_GPU

    @property
    def xp(self):
        # la « librairie » active
        return _cp if self.use_gpu else _np

    def asarray(self, x):
        """Convertit n'importe quel objet en xp.ndarray (CPU ou GPU) sans bug."""
        if self.use_gpu:
            if isinstance(x, _cp.ndarray):
                return x
            elif isinstance(x, _np.ndarray):
                return _cp.asarray(x)
            else:
                return _cp.asarray(x)
        else:
            if isinstance(x, _np.ndarray):
                return x
            elif isinstance(x, _cp.ndarray):
                return _cp.asnumpy(x)
            else:
                return _np.asarray(x)

    def to_cpu(self, x):
        # renvoie un np.ndarray
        return _cp.asnumpy(x) if self.use_gpu else x

    def to_gpu(self, x):
        # renvoie un cp.ndarray
        return _cp.asarray(x) if not self.use_gpu else x
    
    def set_gpu(self, flag: bool):
        """Active (ou désactive) le GPU à la volée."""
        self.use_gpu = bool(flag)
        os.environ["USE_GPU"] = "1" if flag else "0"

# instance singleton
backend = Backend()

class HasBackend:
    @property
    def xp(self):
        return backend.xp
    
    def asarray(self, x):
        return backend.asarray(x)