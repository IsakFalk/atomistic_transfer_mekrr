from abc import ABC, abstractmethod

import torch
import torch.linalg as LA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from torch import Tensor


# Kernels
class Kernel(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass


class MeanEmbeddingKernel(ABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, x, y):
        pass

    def _mean_embedding_kernel(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        t, n, d = x.shape
        l, m, d = y.shape  # noqa
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y).reshape(t, n, l, m)  # t x n x l x m
        # to get the actual embedding kernel we have to sum over the point cloud axes
        k = k0.sum(axis=(1, 3)) / (n * m)
        return k


class GroupEmbeddingKernel(ABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, x, y, **kwargs):
        pass

    def _group_embedding_kernel(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        assert x.shape[:2] == freq_x.shape
        assert y.shape[:2] == freq_y.shape
        t, n, d = x.shape
        l, m, d = y.shape  # noqa
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y)  # t*n x l*m
        # We weigh each point by the inverse of the number of atoms of
        # that species in each frame
        freq_x = freq_x.reshape(t * n)
        freq_y = freq_y.reshape(l * m)
        k0 = k0 * torch.outer(freq_x, freq_y)
        k = k0.reshape(t, n, l, m).sum(axis=(1, 3))
        return k


class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        D2 = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-D2 / (2 * self.sigma**2))


class LinearKernel(Kernel):
    def __init__(self):
        pass

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.matmul(x, y.T)


# Kernel Mean Embeddings
class LinearMeanEmbeddingKernel(MeanEmbeddingKernel):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self._mean_embedding_kernel(x, y)


class GaussianMeanEmbeddingKernel(MeanEmbeddingKernel):
    def __init__(self, kernel: Kernel, sigma=1.0):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        kxx = self._mean_embedding_kernel(x, x).diag().reshape(-1, 1)
        kyy = self._mean_embedding_kernel(y, y).diag().reshape(1, -1)
        kxy = self._mean_embedding_kernel(x, y)
        k = kxx + kyy - 2 * kxy  # this is like ||x-y||^2 vectorized
        return torch.exp(-k / (2 * self.sigma**2))


# Group Kernel Embeddings
class LinearGroupEmbeddingKernel(GroupEmbeddingKernel):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        return self._group_embedding_kernel(x, y, freq_x, freq_y)


class GaussianGroupEmbeddingKernel(GroupEmbeddingKernel):
    def __init__(self, kernel: Kernel, sigma=1.0):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        kxx = self._group_embedding_kernel(x, x, freq_x, freq_x).diag().reshape(-1, 1)
        kyy = self._group_embedding_kernel(y, y, freq_y, freq_y).diag().reshape(1, -1)
        kxy = self._group_embedding_kernel(x, y, freq_x, freq_y)
        k = kxx + kyy - 2 * kxy  # this is like ||x-y||^2 vectorized
        return torch.exp(-k / (2 * self.sigma**2))


class EmbeddingKernel:
    def __init__(self, kernel: Kernel, alpha: int = 0.0, aggregation: str = "mean"):
        self.kernel = kernel
        self.alpha = alpha
        self.aggregation = aggregation  # dispatch on this: mean, sum

    def __call__(self, x: Tensor, zx: Tensor, y: Tensor, zy: Tensor) -> Tensor:
        return self._embedding_kernel(x, zx, y, zy)

    def _embedding_kernel(self, x: Tensor, zx: Tensor, y: Tensor, zy: Tensor) -> Tensor:
        t, n, d = x.shape
        l, m, d = y.shape  # noqa
        assert zx.shape[:2] == (t, n)
        assert zy.shape[:2] == (l, m)
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y)
        # The embedding kernel is a (1-a) * agg_c * K + a * group_c * K

        # Linear algebra magic: We make the mask needed for the group embedding kernel
        # essentially we vectorize the coefficient matrix which for each pair x_t, y_l of systems
        # has value B_ij = (alpha - 1) * c0^2 + alpha * delta(zx_i, zy_j) c_zx_i * c_zy_j
        # where c_zx_i and c_zy_j are the number of points in the point clouds of x_t and y_l if we do mean embedding
        # or just 1 if we do sum embedding
        zx = zx.reshape(t * n, -1)
        zy = zy.reshape(l * m, -1)

        delta = (zx[:, None] == zy[None, :]).squeeze()

        agg_c = 1.0
        if self.aggregation == "mean":
            agg_c /= n * m
            # Get all groups and possibly calculate the number
            groups = torch.unique(torch.cat([zx, zy]))
            group_nx = (zx[:, None] == groups[None, :]).reshape(t, n, -1).sum(axis=1)
            group_ny = (zy[:, None] == groups[None, :]).reshape(l, m, -1).sum(axis=1)
            select_zx = (zx[:, None] == groups[None, :]).reshape(t, n, -1)
            select_zy = (zy[:, None] == groups[None, :]).reshape(l, m, -1)
            group_nx_flatten = (select_zx * group_nx[:, None, :]).reshape(t * n, -1).sum(axis=1)
            group_ny_flatten = (select_zy * group_ny[:, None, :]).reshape(l * m, -1).sum(axis=1)
            # Normalize by the number of kernels in the group \sum_s K_s / S
            mask = (1.0 / group_nx_flatten[:, None]) * (1.0 / group_ny_flatten[None, :])
        elif self.aggregation == "sum":
            mask = 1.0
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation}")
        group_c = mask * delta

        k = (k0 * ((1 - self.alpha) * agg_c + self.alpha * group_c)).reshape(t, n, l, m).sum(axis=(1, 3))
        return k


# Estimators
### Create distribution regression
### Output is assumed to be in \R and
### we have T snapshots, indexed by t, and the system is
### described by N atoms, indexed by i.
### This means that the kernel (when using some pointwise kernel K)
### is of size T x T
### Call this gram matrix K, then
### K_{t, l} = torch.sum(G^{t, l}) / N**2 where
### G^{t, l}_{i, j} = K(x^{t}_{i}, x^{l}_{j})
### Thus, the only thing we need to do is to build a kernel tensor
### of size T x T x N x N and then sum over the last two dimensions
### (or equivalently)


class KernelMeanEmbeddingRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel: MeanEmbeddingKernel, lmbda: float = 1.0):
        self.kernel = kernel
        self.lmbda = lmbda

    def fit(self, X: Tensor, y: Tensor):
        assert len(X.shape) == 3
        assert len(y.shape) == 2
        self.X_ = X
        self.y_ = y

        k = self.kernel(X, X)
        # klmbda = k + torch.eye(k.shape[0]) * self.lmbda
        # Below is the same as above but avoids the creation of a new tensor on a different device
        klmbda = (k - k.diag().diag()) + (self.lmbda * self.X_.shape[0] + k.diag()).diag()
        self.k_ = k
        self.alpha_ = LA.solve(klmbda, y)
        return self

    def predict(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 3
        k = self.kernel(X, self.X_)
        return k @ self.alpha_

    def predict_y_and_grad(self, X: Tensor, pos: Tensor) -> Tensor:
        assert len(X.shape) == 3
        T, num_atoms, d = X.shape
        # self.X_.requires_grad_(True)
        k = self.kernel(X, self.X_)
        y_pred = k @ self.alpha_
        grad_pred = (
            torch.autograd.grad(
                y_pred,
                pos,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=False,
                allow_unused=False,
            )[0]
        ).reshape(T, num_atoms, -1)
        return y_pred, grad_pred

    def score(self, X: Tensor, y: Tensor) -> float:
        y_pred = self.predict(X)
        return float(mean_squared_error(y, y_pred))


class KernelGroupEmbeddingRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel: GroupEmbeddingKernel, lmbda: float = 1.0):
        self.kernel = kernel
        self.lmbda = lmbda

    def fit(self, X: Tensor, y: Tensor, freq: Tensor):
        assert X.shape[:2] == freq.shape[:2]
        assert len(X.shape) == 3
        assert len(y.shape) == 2
        self.X_ = X
        self.y_ = y
        self._freq = freq

        k = self.kernel(X, X, freq, freq)
        # klmbda = k + torch.eye(k.shape[0]) * self.lmbda
        # Below is the same as above but avoids the creation of a new tensor on a different device
        klmbda = (k - k.diag().diag()) + (self.lmbda * self.X_.shape[0] + k.diag()).diag()
        self.k_ = k
        self.alpha_ = LA.solve(klmbda, y)
        return self

    def predict(self, X: Tensor, freq: Tensor) -> Tensor:
        assert len(X.shape) == 3
        assert X.shape[:2] == freq.shape[:2]
        k = self.kernel(X, self.X_, freq, self._freq)
        return k @ self.alpha_

    def predict_y_and_grad(self, X: Tensor, pos: Tensor, freq: Tensor) -> Tensor:
        assert len(X.shape) == 3
        assert X.shape[:2] == freq.shape[:2]
        T, num_atoms, d = X.shape
        # self.X_.requires_grad = True # NOTE: This may mess with torch functional, thing about forcing this outside
        k = self.kernel(X, self.X_, freq, self._freq)
        y_pred = k @ self.alpha_
        grad_pred = (
            torch.autograd.grad(
                y_pred,
                pos,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=False,
                allow_unused=False,
            )[0]
        ).reshape(T, num_atoms, -1)
        return y_pred, grad_pred

    def score(self, X: Tensor, y: Tensor, freq: Tensor) -> float:
        y_pred = self.predict(X, freq)
        return float(mean_squared_error(y, y_pred))


def median_heuristic(x: Tensor, y: Tensor) -> float:
    return float(torch.median(torch.cdist(x, y, p=2)))


# Sklearn utilities
class TorchStandardScaler:
    """Standardization for torch"""

    def __init__(self, eps=1e-7):
        self.eps = eps

    def fit(self, x):
        self.mean_ = x.mean(0, keepdim=True)
        self.std_ = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean_
        x /= self.std_ + self.eps
        return x

    def inverse_transform(self, x):
        x *= self.std_ + self.eps
        x += self.mean_
        return x


class StandardizedOutputRegression(BaseEstimator, RegressorMixin):
    """Wrapper class which standardizes the output of a univariate regression model (torch)

    Parameters:
        ------------
        regressor: object
            The regression model to be wrapped.

        eps: float, default=1e-7
            A small number to be added to the standard deviation when dividing to avoid division
            by zero.

    Methods:
        ---------
        fit(self, X, y):
            Fit the standardized regression model to the training data.

        predict(self, X):
            Predict using the standardized regression model.

        Returns: y_pred
        --------
    """

    def __init__(self, regressor, eps=1e-7):
        self.regressor = regressor
        self.scaler = TorchStandardScaler(eps)

    def fit(self, X, y):
        assert len(y.shape) == 2 and y.shape[1] == 1, "y should have shape 2 and be univariate"
        self.scaler.fit(y)
        y = self.scaler.transform(y)
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.scaler.inverse_transform(self.regressor.predict(X))

    def predict_y_and_grad(self, X: Tensor, pos: Tensor) -> Tensor:
        y_pred, grad_pred = self.regressor.predict_y_and_grad(X, pos)
        y_pred = self.scaler.inverse_transform(y_pred)
        grad_pred = self.scaler.std_.item() * grad_pred
        return y_pred, grad_pred


class EmbeddingRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel: EmbeddingKernel, lmbda: float = 1.0):
        self.kernel = kernel
        self.lmbda = lmbda

    def fit(self, X: Tensor, ZX: Tensor, y: Tensor):
        assert len(X.shape) == 3
        assert len(ZX.shape) == 3
        assert len(y.shape) == 2
        self.X_ = X
        self.ZX_ = ZX
        self.y_ = y

        k = self.kernel(X, ZX, X, ZX)
        # klmbda = k + torch.eye(k.shape[0]) * self.lmbda
        # Below is the same as above but avoids the creation of a new tensor on a different device
        klmbda = (k - k.diag().diag()) + (self.lmbda * self.X_.shape[0] + k.diag()).diag()
        self.k_ = k
        self.alpha_ = LA.solve(klmbda, y)
        return self

    def predict(self, X: Tensor, ZX: Tensor) -> Tensor:
        assert len(X.shape) == 3
        assert len(ZX.shape) == 3
        k = self.kernel(X, ZX, self.X_, self.ZX_)
        return k @ self.alpha_

    def predict_y_and_grad(self, X: Tensor, ZX: Tensor, pos: Tensor) -> Tensor:
        assert len(X.shape) == 3
        T, num_atoms, d = X.shape
        # self.X_.requires_grad_(True)
        k = self.kernel(X, ZX, self.X_, self.ZX_)
        y_pred = k @ self.alpha_
        grad_pred = (
            torch.autograd.grad(
                y_pred,
                pos,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=False,
                allow_unused=False,
            )[0]
        ).reshape(T, num_atoms, -1)
        return y_pred, grad_pred

    def score(self, X: Tensor, ZX: Tensor, y: Tensor) -> float:
        y_pred = self.predict(X, ZX)
        return float(mean_squared_error(y, y_pred))
