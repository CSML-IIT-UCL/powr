import jax.numpy as jnp


class Qmodel:
    def __init__(self, kernel=None, Q=None, X_sub=None):

        assert kernel is not None
        assert X_sub is not None
        assert Q is not None

        self.kernel = kernel
        self.Q = Q
        self.X_sub = X_sub

        # Pre-compute and cache the RFF transform of X_sub if kernel supports it
        self._cached_Zy = None
        if hasattr(kernel, '_transform'):
            self._cached_Zy = kernel._transform(X_sub)

    def evaluate(self, X=None):
        # Use cached transform if available (for RFF kernels)
        if self._cached_Zy is not None and hasattr(self.kernel, '_transform'):
            Zx = self.kernel._transform(X)
            return (Zx @ self._cached_Zy.T) @ self.Q
        return self.kernel(X, self.X_sub) @ self.Q

    def __getstate__(self):
        """ Prepare the object for pickling by returning only necessary attributes. """
        state = {
            'Q': self.Q,
            'X_sub': self.X_sub,
            '_cached_Zy': self._cached_Zy,
        }
        return state

    def __setstate__(self, state):
        """ Restore the object's state with only the necessary attributes. """
        self.Q = state['Q']
        self.X_sub = state['X_sub']
        self._cached_Zy = state.get('_cached_Zy', None)