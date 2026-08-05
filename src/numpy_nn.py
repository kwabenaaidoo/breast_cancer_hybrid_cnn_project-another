"""
numpy_nn.py
------------
A small, dependency-free (NumPy-only) neural network toolkit: Conv1D,
MaxPool1D, Dense, ReLU, Dropout layers with manual forward/backward passes,
plus an Adam optimizer. This exists so the project has ZERO dependency on
TensorFlow/PyTorch -- numpy has universal prebuilt wheels for every Python
version, OS, and CPU architecture, which avoids the class of installation
failures those larger frameworks are prone to (missing wheels for new
Python releases, multi-hundred-MB downloads timing out on restricted
networks, compiler toolchain requirements when a wheel isn't available).

Every layer follows the same contract:
    forward(x, training)  -> output   (and stores what backward() needs)
    backward(dout)         -> dx        (and stores parameter gradients)

Shapes follow the "channels-last" convention used by Keras, so this module
is a drop-in conceptual replacement for the CNN architecture originally
described in the brief:
    Conv1D -> ReLU -> MaxPool1D -> Conv1D -> ReLU -> MaxPool1D
    -> Flatten -> Dense -> ReLU -> Dropout -> Dense(feature layer) -> ReLU
    -> Dropout -> Dense(1) -> Sigmoid
"""

import math
import numpy as np


# ---------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------
class Conv1D:
    """1D convolution, stride 1, 'same' padding (output length == input length)."""

    def __init__(self, in_channels, out_channels, kernel_size, rng):
        fan_in = in_channels * kernel_size
        std = math.sqrt(2.0 / fan_in)  # He initialization (ReLU-friendly)
        self.W = rng.normal(0, std, size=(kernel_size, in_channels, out_channels))
        self.b = np.zeros(out_channels)
        self.K = kernel_size
        self._cache = None
        self.dW = None
        self.db = None

    def forward(self, x, training=True):
        # x: (N, L, C_in)
        N, L, C_in = x.shape
        K = self.K
        pad_left = (K - 1) // 2
        pad_right = (K - 1) - pad_left
        x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))
        L_out = L  # 'same' padding, stride 1

        out = np.zeros((N, L_out, self.W.shape[2]))
        for k in range(K):
            out += x_padded[:, k:k + L_out, :] @ self.W[k]
        out += self.b

        self._cache = (x_padded, pad_left, L)
        return out

    def backward(self, dout):
        x_padded, pad_left, L = self._cache
        K = self.K
        L_out = dout.shape[1]

        self.dW = np.zeros_like(self.W)
        dx_padded = np.zeros_like(x_padded)
        for k in range(K):
            xk = x_padded[:, k:k + L_out, :]                     # (N, L_out, C_in)
            self.dW[k] = np.einsum('nlc,nlo->co', xk, dout)      # (C_in, C_out)
            dx_padded[:, k:k + L_out, :] += dout @ self.W[k].T   # (N, L_out, C_in)
        self.db = dout.sum(axis=(0, 1))

        dx = dx_padded[:, pad_left:pad_left + L, :]
        return dx

    def params_and_grads(self, prefix):
        return {f"{prefix}_W": (self.W, self.dW), f"{prefix}_b": (self.b, self.db)}


class MaxPool1D:
    """Max pooling, 'same'-style padding with -inf so padding never wins the max."""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self._cache = None

    def forward(self, x, training=True):
        N, L, C = x.shape
        p = self.pool_size
        L_out = math.ceil(L / p)
        pad_len = L_out * p - L
        if pad_len > 0:
            x_padded = np.pad(x, ((0, 0), (0, pad_len), (0, 0)),
                               mode="constant", constant_values=-np.inf)
        else:
            x_padded = x

        x_reshaped = x_padded.reshape(N, L_out, p, C)
        out = x_reshaped.max(axis=2)
        argmax = x_reshaped.argmax(axis=2)  # (N, L_out, C), values in [0, p)

        self._cache = (x.shape, pad_len, argmax, p)
        return out

    def backward(self, dout):
        orig_shape, pad_len, argmax, p = self._cache
        N, L_out, C = dout.shape

        dx_padded = np.zeros((N, L_out * p, C))
        n_idx, l_idx, c_idx = np.meshgrid(
            np.arange(N), np.arange(L_out), np.arange(C), indexing="ij")
        pos = l_idx * p + argmax
        dx_padded[n_idx, pos, c_idx] = dout

        L = orig_shape[1]
        return dx_padded[:, :L, :]


class Flatten:
    def __init__(self):
        self._shape = None

    def forward(self, x, training=True):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dense:
    def __init__(self, in_features, out_features, rng):
        std = math.sqrt(2.0 / in_features)
        self.W = rng.normal(0, std, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self._cache = None
        self.dW = None
        self.db = None

    def forward(self, x, training=True):
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self._cache
        self.dW = x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self, prefix):
        return {f"{prefix}_W": (self.W, self.dW), f"{prefix}_b": (self.b, self.db)}


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x, training=True):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Dropout:
    def __init__(self, rate, rng):
        self.rate = rate
        self.rng = rng
        self._mask = None

    def forward(self, x, training=True):
        if training and self.rate > 0:
            self._mask = (self.rng.random(x.shape) >= self.rate) / (1 - self.rate)
            return x * self._mask
        self._mask = None
        return x

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------
class Adam:
    """Standard Adam optimizer operating over a dict of {name: (param, grad)}."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_and_grads: dict):
        self.t += 1
        for name, (param, grad) in params_and_grads.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
