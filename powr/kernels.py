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
        # Gaussian/RBF kernel: exp(-||x-y||^2 / (2*sigma^2))
        diff = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)
        sq_dist = jnp.sum(diff ** 2, axis=2)
        return jnp.exp(-sq_dist / (2 * self.sigma ** 2))


# gaussian kernel for matrices of n points and d dimension with a different sigma for each dimension
class gaussian_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
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
    """Random Fourier Features for Gaussian/RBF kernel (uses Gaussian spectral distribution)."""
    def __init__(self, sigma, n_features=256, seed=0):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        self.n_features = int(n_features)
        self._key = random.PRNGKey(int(seed))
        self.omega = None   # shape (d, D)
        self.b = None       # shape (D,)
        self.scaling = jnp.sqrt(2.0 / self.n_features)

        # Caching for transformed features
        self._cache_Y = None  # Cached Y input (for identity check)
        self._cache_Zy = None  # Cached transform of Y
        self._cache_enabled = True

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

    def cache_points(self, Y):
        """
        Explicitly cache the RFF transform of a set of points (e.g., X_sub).
        This is useful when Y is fixed and used repeatedly as the second argument.

        Args:
            Y: Points to cache, shape (m, d)
        """
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        self._cache_Y = Y
        self._cache_Zy = self._transform(Y)

    def clear_cache(self):
        """Clear the cached transform."""
        self._cache_Y = None
        self._cache_Zy = None

    def _is_cached(self, Y):
        """Check if Y matches the cached points using identity/shape check."""
        if self._cache_Y is None or self._cache_Zy is None:
            return False
        # Fast check: same object reference or same shape and data pointer
        if self._cache_Y is Y:
            return True
        # Shape check
        if self._cache_Y.shape != Y.shape:
            return False
        # For JAX arrays, check if they share the same underlying data
        # This avoids expensive element-wise comparison
        try:
            # Use pointer comparison for efficiency
            return self._cache_Y.unsafe_buffer_pointer() == Y.unsafe_buffer_pointer()
        except (AttributeError, TypeError):
            # Fallback: check if arrays are identical (same memory)
            return self._cache_Y is Y

    def __call__(self, X, Y):
        Zx = self._transform(X)

        # Check if Y is cached
        if self._cache_enabled and self._is_cached(Y):
            Zy = self._cache_Zy
        else:
            Zy = self._transform(Y)
            # Auto-cache Y if caching is enabled (useful for repeated calls with same Y)
            if self._cache_enabled:
                self._cache_Y = Y
                self._cache_Zy = Zy

        return Zx @ Zy.T

class RFFLaplace:
    """
    Random Fourier Features for Laplace kernel (uses Cauchy spectral distribution).
    
    The Laplace kernel k(x,y) = exp(-||x-y||/sigma) has a Cauchy spectral density,
    so we sample omega from Cauchy distribution instead of Gaussian.
    """
    def __init__(self, sigma, n_features=256, seed=0):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        self.n_features = int(n_features)
        self._key = random.PRNGKey(int(seed))
        self.omega = None   # shape (d, D)
        self.b = None       # shape (D,)
        self.scaling = jnp.sqrt(2.0 / self.n_features)

        # Caching for transformed features
        self._cache_Y = None
        self._cache_Zy = None
        self._cache_enabled = True

    def _initialize(self, dim):
        k1, k2 = random.split(self._key)
        # Sample from Cauchy distribution for Laplace kernel
        # Cauchy can be generated as: tan(pi * (U - 0.5)) where U ~ Uniform(0,1)
        u = random.uniform(k1, (dim, self.n_features), minval=1e-6, maxval=1.0 - 1e-6)
        omega_cauchy = jnp.tan(jnp.pi * (u - 0.5))
        
        # Scale by sigma
        if self.sigma.size == 1:
            omega = omega_cauchy / self.sigma.astype(jnp.float64)
        else:
            s = jnp.asarray(self.sigma, dtype=jnp.float64).reshape(-1, 1)
            if s.shape[0] != dim:
                raise ValueError("sigma length does not match input dim")
            omega = omega_cauchy / s
            
        b = random.uniform(k2, (self.n_features,), minval=0.0, maxval=2.0 * jnp.pi)
        self.omega = jnp.array(omega)
        self.b = jnp.array(b)

    def _transform(self, X):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        if self.omega is None:
            self._initialize(X.shape[-1])
        proj = X @ self.omega + self.b  # (n, D)
        return self.scaling * jnp.cos(proj)

    def cache_points(self, Y):
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        self._cache_Y = Y
        self._cache_Zy = self._transform(Y)

    def clear_cache(self):
        self._cache_Y = None
        self._cache_Zy = None

    def _is_cached(self, Y):
        if self._cache_Y is None or self._cache_Zy is None:
            return False
        if self._cache_Y is Y:
            return True
        if self._cache_Y.shape != Y.shape:
            return False
        try:
            return self._cache_Y.unsafe_buffer_pointer() == Y.unsafe_buffer_pointer()
        except (AttributeError, TypeError):
            return self._cache_Y is Y

    def __call__(self, X, Y):
        Zx = self._transform(X)
        if self._cache_enabled and self._is_cached(Y):
            Zy = self._cache_Zy
        else:
            Zy = self._transform(Y)
            if self._cache_enabled:
                self._cache_Y = Y
                self._cache_Zy = Zy
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


# ============================================================================
# Matern Kernels
# ============================================================================

class matern_kernel_12:
    """
    Matern kernel with nu=1/2 (equivalent to Laplace/exponential kernel).
    
    k(x, y) = exp(-||x - y|| / sigma)
    
    This is the least smooth Matern kernel.
    """
    def __init__(self, sigma):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        
        if self.sigma.size == 1:
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
            return jnp.exp(-dist / self.sigma)
        else:
            # Per-dimension sigma
            scaled_diff = diff / self.sigma.reshape(1, 1, -1)
            dist = jnp.sqrt(jnp.sum(scaled_diff ** 2, axis=-1))
            return jnp.exp(-dist)


class matern_kernel_32:
    """
    Matern kernel with nu=3/2.
    
    k(x, y) = (1 + sqrt(3) * d / sigma) * exp(-sqrt(3) * d / sigma)
    
    where d = ||x - y||
    
    This produces once-differentiable functions.
    """
    def __init__(self, sigma):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        self.sqrt3 = jnp.sqrt(3.0)
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        
        if self.sigma.size == 1:
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
            scaled_dist = self.sqrt3 * dist / self.sigma
        else:
            # Per-dimension sigma
            scaled_diff = diff / self.sigma.reshape(1, 1, -1)
            dist = jnp.sqrt(jnp.sum(scaled_diff ** 2, axis=-1))
            scaled_dist = self.sqrt3 * dist
            
        return (1.0 + scaled_dist) * jnp.exp(-scaled_dist)


class matern_kernel_52:
    """
    Matern kernel with nu=5/2.
    
    k(x, y) = (1 + sqrt(5) * d / sigma + 5 * d^2 / (3 * sigma^2)) * exp(-sqrt(5) * d / sigma)
    
    where d = ||x - y||
    
    This produces twice-differentiable functions.
    """
    def __init__(self, sigma):
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64)
        self.sqrt5 = jnp.sqrt(5.0)
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        
        if self.sigma.size == 1:
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
            scaled_dist = self.sqrt5 * dist / self.sigma
            scaled_dist_sq = 5.0 * (dist ** 2) / (3.0 * self.sigma ** 2)
        else:
            # Per-dimension sigma
            scaled_diff = diff / self.sigma.reshape(1, 1, -1)
            dist = jnp.sqrt(jnp.sum(scaled_diff ** 2, axis=-1))
            scaled_dist = self.sqrt5 * dist
            scaled_dist_sq = (5.0 / 3.0) * (dist ** 2)
            
        return (1.0 + scaled_dist + scaled_dist_sq) * jnp.exp(-scaled_dist)


class matern_kernel_diag:
    """
    Generic Matern kernel with per-dimension sigma.
    
    Args:
        sigma: Bandwidth parameter(s)
        nu: Smoothness parameter (0.5, 1.5, or 2.5)
    """
    def __init__(self, sigma, nu=1.5):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)
        self.nu = nu
        
        if nu == 0.5:
            self._kernel_fn = self._matern_12
        elif nu == 1.5:
            self._kernel_fn = self._matern_32
        elif nu == 2.5:
            self._kernel_fn = self._matern_52
        else:
            raise ValueError(f"Unsupported nu={nu}. Use 0.5, 1.5, or 2.5")
            
    def _matern_12(self, dist):
        return jnp.exp(-dist)
        
    def _matern_32(self, dist):
        sqrt3 = jnp.sqrt(3.0)
        scaled = sqrt3 * dist
        return (1.0 + scaled) * jnp.exp(-scaled)
        
    def _matern_52(self, dist):
        sqrt5 = jnp.sqrt(5.0)
        scaled = sqrt5 * dist
        return (1.0 + scaled + (scaled ** 2) / 3.0) * jnp.exp(-scaled)
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        
        diff = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)
        scaled_diff = diff / self.sigma
        dist = jnp.sqrt(jnp.sum(scaled_diff ** 2, axis=2))
        
        return self._kernel_fn(dist)


# ============================================================================
# Polynomial Kernel
# ============================================================================

class polynomial_kernel:
    """
    Polynomial kernel: k(x, y) = (x^T y + c)^d
    
    Args:
        degree: Polynomial degree
        c: Constant term (default 1.0)
    """
    def __init__(self, degree=2, c=1.0):
        self.degree = degree
        self.c = c
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        
        # Compute inner products
        inner = X @ Y.T + self.c
        
        return inner ** self.degree


class polynomial_kernel_normalized:
    """
    Normalized polynomial kernel: k(x, y) = ((x^T y + c) / (||x|| ||y|| + c))^d
    
    This provides better numerical stability.
    """
    def __init__(self, degree=2, c=1.0):
        self.degree = degree
        self.c = c
        
    def __call__(self, X, Y):
        X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
        Y = jnp.atleast_2d(jnp.asarray(Y, dtype=jnp.float64))
        
        # Compute norms
        norm_X = jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        norm_Y = jnp.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
        
        # Normalize
        X_norm = X / norm_X
        Y_norm = Y / norm_Y
        
        # Compute kernel
        inner = X_norm @ Y_norm.T + self.c
        
        return inner ** self.degree


# ============================================================================
# Kernel Factory
# ============================================================================

def create_kernel(kernel_type: str, **kwargs):
    """
    Factory function to create kernels by name.
    
    Args:
        kernel_type: One of "gaussian", "laplace", "matern12", "matern32", 
                     "matern52", "polynomial", "rff"
        **kwargs: Kernel-specific parameters
        
    Returns:
        Kernel function
    """
    kernel_type = kernel_type.lower()
    
    if kernel_type in ("gaussian", "rbf", "exact"):
        sigma = kwargs.get("sigma", 1.0)
        return gaussian_kernel(sigma, method="exact")
        
    elif kernel_type == "rff":
        sigma = kwargs.get("sigma", 1.0)
        n_features = kwargs.get("n_features", 256)
        seed = kwargs.get("seed", 0)
        return RFFGaussian(sigma=sigma, n_features=n_features, seed=seed)
        
    elif kernel_type in ("rff_laplace", "rff_cauchy"):
        sigma = kwargs.get("sigma", 1.0)
        n_features = kwargs.get("n_features", 256)
        seed = kwargs.get("seed", 0)
        return RFFLaplace(sigma=sigma, n_features=n_features, seed=seed)
        
    elif kernel_type in ("laplace", "abel", "matern12"):
        sigma = kwargs.get("sigma", 1.0)
        if hasattr(sigma, '__len__'):
            return abel_kernel_diag(sigma)
        return matern_kernel_12(sigma)
        
    elif kernel_type == "matern32":
        sigma = kwargs.get("sigma", 1.0)
        if hasattr(sigma, '__len__'):
            return matern_kernel_diag(sigma, nu=1.5)
        return matern_kernel_32(sigma)
        
    elif kernel_type == "matern52":
        sigma = kwargs.get("sigma", 1.0)
        if hasattr(sigma, '__len__'):
            return matern_kernel_diag(sigma, nu=2.5)
        return matern_kernel_52(sigma)
        
    elif kernel_type == "polynomial":
        degree = kwargs.get("degree", 2)
        c = kwargs.get("c", 1.0)
        normalized = kwargs.get("normalized", False)
        if normalized:
            return polynomial_kernel_normalized(degree, c)
        return polynomial_kernel(degree, c)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")