"""
Stub Torch Module

Provides basic tensor operations when real torch is unavailable.
Used when network is down or torch installation is incomplete.

This is NOT a replacement for real torch - just enough for the system to boot.
"""

import numpy as np
from typing import Any, List, Optional, Tuple

# Flag to indicate this is a stub
IS_STUB = True


class StubTensor:
    """Stub tensor class that wraps numpy."""
    
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        self._data = np.array(data, dtype=dtype)
        self.dtype = self._data.dtype
        self.device = device or 'cpu'
        self.requires_grad = requires_grad
        self.grad = None
        self._shape = self._data.shape
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def ndim(self) -> int:
        return len(self._shape)
    
    def numpy(self) -> np.ndarray:
        return self._data.copy()
    
    def cpu(self):
        return self
    
    def cuda(self):
        raise RuntimeError("CUDA not available (stub torch)")
    
    def to(self, device):
        if device != 'cpu':
            raise RuntimeError(f"Device {device} not available (stub torch)")
        return self
    
    def backward(self):
        pass  # No gradients in stub
    
    def __repr__(self):
        return f"StubTensor({self._data}, dtype={self.dtype})"
    
    def __add__(self, other):
        if isinstance(other, StubTensor):
            return StubTensor(self._data + other._data)
        return StubTensor(self._data + other)
    
    def __sub__(self, other):
        if isinstance(other, StubTensor):
            return StubTensor(self._data - other._data)
        return StubTensor(self._data - other)
    
    def __mul__(self, other):
        if isinstance(other, StubTensor):
            return StubTensor(self._data * other._data)
        return StubTensor(self._data * other)


def tensor(data, dtype=None, device=None, requires_grad=False) -> StubTensor:
    return StubTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=None, device=None) -> StubTensor:
    return StubTensor(np.zeros(shape), dtype=dtype, device=device)


def ones(*shape, dtype=None, device=None) -> StubTensor:
    return StubTensor(np.ones(shape), dtype=dtype, device=device)


def randn(*shape, dtype=None, device=None) -> StubTensor:
    return StubTensor(np.random.randn(*shape), dtype=dtype, device=device)


def randint(low, high, *shape, dtype=None, device=None) -> StubTensor:
    return StubTensor(np.random.randint(low, high, shape), dtype=dtype, device=device)


class StubModule:
    """Stub nn.Module."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def parameters(self):
        return list(self._parameters.values())
    
    def named_parameters(self):
        return [(k, v) for k, v in self._parameters.items()]
    
    def register_parameter(self, name, param):
        self._parameters[name] = param
    
    def register_module(self, name, module):
        self._modules[name] = module
    
    def __setattr__(self, name, value):
        if isinstance(value, (StubTensor, StubModule)):
            if isinstance(value, StubTensor):
                self.register_parameter(name, value)
            else:
                self.register_module(name, value)
        super().__setattr__(name, value)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Stub module - implement forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Provide basic nn functions
class nn:
    """Stub nn module."""
    
    class Module(StubModule):
        pass
    
    class Linear(StubModule):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = StubTensor(np.random.randn(out_features, in_features) * 0.01)
            self.bias = StubTensor(np.zeros(out_features)) if bias else None
        
        def forward(self, x):
            if isinstance(x, StubTensor):
                x = x._data
            result = np.dot(x, self.weight._data.T)
            if self.bias is not None:
                result += self.bias._data
            return StubTensor(result)
    
    class Embedding(StubModule):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = StubTensor(np.random.randn(num_embeddings, embedding_dim) * 0.01)
        
        def forward(self, x):
            if isinstance(x, StubTensor):
                x = x._data
            indices = x.flatten()
            result = self.weight._data[indices]
            return StubTensor(result.reshape(*x.shape, -1))
    
    class ReLU(StubModule):
        def forward(self, x):
            if isinstance(x, StubTensor):
                data = np.maximum(0, x._data)
            else:
                data = np.maximum(0, x)
            return StubTensor(data)
    
    class Sigmoid(StubModule):
        def forward(self, x):
            if isinstance(x, StubTensor):
                data = 1 / (1 + np.exp(-x._data))
            else:
                data = 1 / (1 + np.exp(-x))
            return StubTensor(data)
    
    class Tanh(StubModule):
        def forward(self, x):
            if isinstance(x, StubTensor):
                data = np.tanh(x._data)
            else:
                data = np.tanh(x)
            return StubTensor(data)
    
    class Dropout(StubModule):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
            self.training = True
        
        def forward(self, x):
            return x  # No dropout in stub
    
    class LayerNorm(StubModule):
        def __init__(self, normalized_shape):
            super().__init__()
            self.weight = StubTensor(np.ones(normalized_shape))
            self.bias = StubTensor(np.zeros(normalized_shape))
        
        def forward(self, x):
            return x
    
    class Sequential(StubModule):
        def __init__(self, *modules):
            super().__init__()
            for i, m in enumerate(modules):
                self.register_module(str(i), m)
            self._layers = list(modules)
        
        def forward(self, x):
            for m in self._layers:
                x = m(x)
            return x


class optim:
    """Stub optim module."""
    
    class Adam:
        def __init__(self, params, lr=0.001):
            self.params = list(params)
            self.lr = lr
        
        def step(self):
            pass
        
        def zero_grad(self):
            pass
    
    class SGD:
        def __init__(self, params, lr=0.01, momentum=0):
            self.params = list(params)
            self.lr = lr
            self.momentum = momentum
        
        def step(self):
            pass
        
        def zero_grad(self):
            pass


# Provide no_grad context
class no_grad:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# Provide device detection (always cpu in stub)
def cuda_is_available():
    return False


def mps_is_available():
    return False


# Export everything
__all__ = [
    'tensor', 'zeros', 'ones', 'randn', 'randint',
    'StubTensor', 'StubModule', 'nn', 'optim',
    'no_grad', 'cuda_is_available', 'mps_is_available',
    'IS_STUB',
]
