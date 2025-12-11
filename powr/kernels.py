import jax
import jax.numpy as jnp
from jax import random

# compute the dirac kernel on batches of states
@jax.jit    
def dirac_kernel(X, Y):
    return ((X.reshape(-1, 1) - Y.reshape(1, -1)) == 0) * 1.0


# gaussian kernel for matrices of n points and d dimensions
class gaussian_kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        print("Guassian kernel called")
        return jnp.exp(
            -(1 / self.sigma)
            * jnp.linalg.norm(
                X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1), axis=2
            )
        )


# gaussian kernel for matrices of n points and d dimension with a different sigma for each dimension
class gaussian_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        print("Guassian diagonal kernel called")
        return jnp.exp(
            -jnp.sum(
                (X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)) ** 2
                / (2 * self.sigma**2),
                axis=2, 
            )
        )


class abel_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(
            -jnp.sum(
                jnp.abs(X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1))
                / (jnp.sqrt(2) * self.sigma),
                axis=2,
            )
        )
    
class softmax:

    def __init__(self):
        pass

    def __call__(self, x):
        return jax.nn.softmax(x, axis=1)

class RFFGaussian:
    def __init__(self, sigma, n_features=256, seed=0):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        self.n_features = int(n_features)
        self._key = random.PRNGKey(int(seed))
        self.omega = None   # shape (d, D)
        self.b = None       # shape (D,)
        self.scaling = jnp.sqrt(2.0 / self.n_features)

    def _initialize(self, dim):
        k1, k2 = random.split(self._key)
        # If scalar sigm, then broadcast and if vector use per-dimension scaling
        if self.sigma.size == 1:
            omega = random.normal(k1, (dim, self.n_features)) / (self.sigma.astype(jnp.float64))
        else:
            s = jnp.asarray(self.sigma, dtype=jnp.float64).reshape(-1, 1)
            if s.shape[0] != dim:
                raise ValueError("sigma length does not match input dim")
            omega = random.normal(k1, (dim, self.n_features)) / s
        b = random.uniform(k2, (self.n_features,), minval=0.0, maxval=2.0 * jnp.pi)
        # store as DeviceArray
        self.omega = jnp.array(omega)
        self.b = jnp.array(b)

    def _transform(self, X):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        if self.omega is None:
            self._initialize(X.shape[-1])
        proj = X @ self.omega + self.b  # (n, D)
        return self.scaling * jnp.cos(proj)

    def __call__(self, X, Y):
        Zx = self._transform(X)
        Zy = self._transform(Y)
        return Zx @ Zy.T

# ---------------------------
# gaussian_kernel factory: preserves existing exact behavior but adds a switch
def gaussian_kernel(sigma, method="exact", n_features=256, seed=0):
    # Exact kernel (keeps previous semantics)
    if method in ("exact", "rbf"):
        sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)

        def _exact(X, Y):
            X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
            Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
            diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
            if sigma_arr.size == 1:
                denom = 2.0 * (sigma_arr ** 2)
            else:
                # per-dim sigma -> scale squared per dimension
                denom = 2.0 * (sigma_arr ** 2)
                # compute weighted squared norm
                sq = (diff ** 2) / denom.reshape(1, 1, -1)
                return jnp.exp(-jnp.sum(sq, axis=-1))
            sqnorm = jnp.sum(diff ** 2, axis=-1)
            return jnp.exp(-sqnorm / denom)

        return _exact

    # RFF approximation
    if method in ("rff", "rff_gaussian", "random"):
        return RFFGaussian(sigma=sigma, n_features=n_features, seed=seed)

    raise ValueError(f"Unknown method={method!r} for gaussian_kernel")
