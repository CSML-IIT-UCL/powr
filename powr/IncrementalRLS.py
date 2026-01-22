import os
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import logging
from typing import Optional, Dict, Any

from powr.kernels import dirac_kernel

class IncrementalRLS:
    def __init__(self, kernel=None, n_actions=None, la=1e-3, n_subsamples=None, early_stopping=None, log_path=None, use_woodbury=True):

        assert kernel is not None
        assert n_actions is not None
        assert n_subsamples is not None

        self.kernel = kernel
        self.n_actions = n_actions
        self.n_subsamples = n_subsamples

        self.la = la

        # reset stuff
        self.n = 0
        self.n_sub = None
        self.n_components = 1000

        self.early_stopping_episodes= early_stopping # TODO: find a new effective strategy to early stop the collection
        self.log_path = log_path
        self.above_threshold_count = 0
        self.stop_collection = False

        # Woodbury incremental update settings
        self.use_woodbury = use_woodbury

        # Metrics tracking
        self.track_metrics = True
        self.last_metrics: Dict[str, Any] = {}
        self._reference_B = None  # Reference operator for error computation
        self._reference_r = None  # Reference reward for error computation

        self.reset()

    # verify that we have not called subsample yet
    def check_subsample(self):
        assert self._SUBSAMPLE_HAS_BEEN_CALLED == False

    def reset(self):

        self.X_sub = None
        self.A_sub = None

        self.K_full_sub = jnp.zeros((0, 0))
        self.K_transitions_sub = jnp.zeros((0, 0))
        self.K_sub_sub = None

        self.sub_indices = None

        self.n_sub = None
        self.n_components = 1000

        self._SUBSAMPLE_HAS_BEEN_CALLED = False

        # Woodbury incremental update state
        self.M_inv = None  # Inverse of (K.T @ K + n * la * K_sub_sub + eps * I)
        self.L_sub_sqrt = None  # Cholesky factor of K_sub_sub for regularization updates
        self._n_at_last_train = 0  # Number of samples at last full train
        self._pending_K_rows = None  # New kernel rows since last train
        self._has_trained = False  # Whether we've done at least one full train

    # collect data -> store in memory the data
    def collect_data(self, A, X, Y_transitions, Y_rewards, above_threshold = False, seed = None):

        if self.early_stopping_episodes is not None:
            if self.stop_collection is False:
                
                if above_threshold:
                    self.above_threshold_count += 1 
                else: 
                    self.above_threshold_count = 0
                print(f"above_threshold_count: {self.above_threshold_count}")
                if self.above_threshold_count >= self.early_stopping_episodes:
                    self.stop_collection = True
                    print("Stop collection - len of Dataset: ", self.n)
                    # save into a file the lenght of self.X
                    with open(f"{self.log_path}/dataset_lenght_{seed}.txt", "w") as f:
                        f.write(f"The dataset has a lenght of {str(self.n)})")

        if self.n == 0:
            self.A = A
            self.X = X
            self.Y_transitions = Y_transitions
            self.Y_rewards = Y_rewards

        else:
            
            if not self.stop_collection:
            # check that the data is provided as a list of arrays one for each possible action
                self.X = jnp.vstack([self.X, X])
                self.Y_transitions = jnp.vstack([self.Y_transitions, Y_transitions])
                self.Y_rewards = jnp.vstack([self.Y_rewards, Y_rewards])
                self.A = jnp.hstack([self.A, A])
                
                if self._SUBSAMPLE_HAS_BEEN_CALLED:
                    self.update_kernels(A, X, Y_transitions)

        self.n = self.X.shape[0]
        

    # update the kernels
    def update_kernels(self, A, X, Y_transitions):
        Knew = self.kernel(jnp.vstack([X, Y_transitions]), self.X_sub)
        new_K_rows = Knew[: X.shape[0]] * dirac_kernel(A, self.A_sub)

        self.K_full_sub = jnp.vstack([self.K_full_sub, new_K_rows])
        self.K_transitions_sub = jnp.vstack(
            [self.K_transitions_sub, Knew[X.shape[0] :]]
        )

        # Track new rows for Woodbury incremental updates
        if self._has_trained and self.use_woodbury:
            if self._pending_K_rows is None:
                self._pending_K_rows = new_K_rows
            else:
                self._pending_K_rows = jnp.vstack([self._pending_K_rows, new_K_rows])

    def _woodbury_rank_k_update(self, U):
        """
        Apply Woodbury identity for rank-k update: M_new = M + U.T @ U

        Using: (A + U.T @ U)^{-1} = A^{-1} - A^{-1} @ U.T @ (I + U @ A^{-1} @ U.T)^{-1} @ U @ A^{-1}

        Args:
            U: Matrix of shape (k, n_sub) representing k new rows

        Returns:
            Updated M_inv
        """
        if self.M_inv is None:
            return None

        # V = U @ M_inv, shape: (k, n_sub)
        V = U @ self.M_inv

        # Capacitance matrix: I + U @ M_inv @ U.T, shape: (k, k)
        capacitance = jnp.eye(U.shape[0]) + U @ V.T

        # Solve capacitance system (small k x k matrix)
        # Using solve instead of inv for numerical stability
        cap_solve = jnp.linalg.solve(capacitance, V)  # (k, n_sub)

        # Update: M_inv_new = M_inv - V.T @ cap_inv @ V = M_inv - V.T @ cap_solve
        self.M_inv = self.M_inv - V.T @ cap_solve

        return self.M_inv

    def _woodbury_regularization_update(self, delta_n):
        """
        Apply Woodbury identity for regularization update: M_new = M + delta_n * la * K_sub_sub

        Since K_sub_sub = L_sub @ L_sub.T (Cholesky), this becomes:
        M_new = M + (sqrt(delta_n * la) * L_sub) @ (sqrt(delta_n * la) * L_sub).T

        Which is a rank-n_sub update that can be done via Woodbury.

        Args:
            delta_n: Change in number of samples (n_new - n_old)
        """
        if self.M_inv is None or self.L_sub_sqrt is None:
            return None

        if delta_n <= 0:
            return self.M_inv

        # U = sqrt(delta_n * la) * L_sub, shape: (n_sub, n_sub)
        scale = jnp.sqrt(delta_n * self.la)
        U = scale * self.L_sub_sqrt

        return self._woodbury_rank_k_update(U)

    def _apply_woodbury_updates(self):
        """
        Apply all pending Woodbury updates (new kernel rows + regularization change).
        """
        if self.M_inv is None or self._pending_K_rows is None:
            return False

        # 1. Apply rank-k update for new kernel rows: M += U.T @ U
        U = self._pending_K_rows  # (k, n_sub)
        self._woodbury_rank_k_update(U)

        # 2. Apply regularization update: M += delta_n * la * K_sub_sub
        delta_n = self.n - self._n_at_last_train
        if delta_n > 0:
            self._woodbury_regularization_update(delta_n)

        # Clear pending rows and update counter
        self._pending_K_rows = None
        self._n_at_last_train = self.n

        return True

    def _solve_with_inverse(self):
        """
        Solve the RLS problem using the stored inverse M_inv.

        Returns (r, B) where:
            r: reward weights
            B: transition operator
        """
        # Compute: W = M_inv @ [K.T, K.T @ Y_rewards]
        rhs = jnp.hstack([self.K_full_sub.T, self.K_full_sub.T @ self.Y_rewards])
        W = self.M_inv @ rhs

        r = W[:, -1].reshape(-1, 1)
        B = W[:, :-1]

        return r, B

    def _full_train(self):
        """
        Perform full training using Cholesky decomposition.
        Also initializes M_inv for subsequent Woodbury updates.
        """
        # Compute the regularized Gram matrix
        M = (self.K_full_sub.T @ self.K_full_sub
             + self.n * self.la * self.K_sub_sub
             + 1e-6 * jnp.eye(self.K_full_sub.shape[1]))

        L = jax.lax.linalg.cholesky(M)

        if jnp.isnan(L).any():
            raise ValueError("Error: NaN in the results of the training for Chol")

        # Solve using Cholesky
        W = jax.lax.linalg.triangular_solve(
            L,
            jax.lax.linalg.triangular_solve(
                L,
                jnp.hstack([self.K_full_sub.T, self.K_full_sub.T @ self.Y_rewards]),
                lower=True,
                left_side=True,
                transpose_a=False,
            ),
            lower=True,
            left_side=True,
            transpose_a=True,
        )

        self.r = W[:, -1].reshape(-1, 1)
        self.B = W[:, :-1]

        # Initialize M_inv for Woodbury updates using Cholesky factors
        # M_inv = L^{-T} @ L^{-1}
        if self.use_woodbury:
            I = jnp.eye(self.n_sub)
            L_inv = jax.lax.linalg.triangular_solve(L, I, lower=True, left_side=True)
            self.M_inv = L_inv.T @ L_inv

            # Store Cholesky factor of K_sub_sub for regularization updates
            # Add small epsilon for numerical stability
            K_sub_reg = self.K_sub_sub + 1e-8 * jnp.eye(self.n_sub)
            self.L_sub_sqrt = jax.lax.linalg.cholesky(K_sub_reg)

            self._n_at_last_train = self.n
            self._pending_K_rows = None
            self._has_trained = True

    def subsample(self):

        self.check_subsample()

        if self.n == 0:
            return

        seed = int.from_bytes(os.urandom(4), "big")
        key = jrandom.PRNGKey(seed)

        # if the number of points is smaller than the number of subsamples, we just use all the points
        self.sub_indices = jnp.arange(self.n)
        if self.n > self.n_subsamples:
            self.sub_indices = jrandom.choice(
                key, int(self.n), (self.n_subsamples,), replace=False
            )
        self.n_sub = self.sub_indices.shape[0]

        self.X_sub = self.X[self.sub_indices]
        self.A_sub = self.A[self.sub_indices]

        self.K_full_sub = jnp.zeros((0, self.n_sub))
        self.K_transitions_sub = jnp.zeros((0, self.n_sub))

        self.update_kernels(self.A, self.X, self.Y_transitions)
        self.K_sub_sub = self.K_full_sub[self.sub_indices]

        # Cache X_sub in kernel if supported (e.g., RFFGaussian)
        if hasattr(self.kernel, 'cache_points'):
            self.kernel.cache_points(self.X_sub)

        self._SUBSAMPLE_HAS_BEEN_CALLED = True

    def train(self):
        train_start_time = time.time()
        
        if not self._SUBSAMPLE_HAS_BEEN_CALLED:
            self.subsample()

        # Compute eigendecomposition of K_sub_sub (needed for Q-value projection)
        eig_start = time.time()
        V, W = jax.lax.linalg.eigh(self.K_sub_sub)
        effective_components = min(self.K_sub_sub.shape[0], self.n_components)
        self.V = V[:, -effective_components : ].T
        eig_time = time.time() - eig_start

        # Decide whether to use Woodbury incremental update or full recomputation
        use_incremental = (
            self.use_woodbury
            and self._has_trained
            and self.M_inv is not None
            and self._pending_K_rows is not None
        )

        solve_start = time.time()
        if use_incremental:
            # Use Woodbury identity for incremental update
            logging.debug(f"Using Woodbury incremental update with {self._pending_K_rows.shape[0]} new rows")
            self._apply_woodbury_updates()
            self.r, self.B = self._solve_with_inverse()
        else:
            # Full training using Cholesky decomposition
            logging.debug("Using full Cholesky training")
            self._full_train()
        solve_time = time.time() - solve_start

        # check if the results contain nan
        if jnp.isnan(self.r).any() or jnp.isnan(self.B).any():
            raise ValueError("Error: NaN in the results of the training")

        total_train_time = time.time() - train_start_time
        
        # Track metrics if enabled
        if self.track_metrics:
            self.last_metrics = {
                'n_samples': self.n,
                'n_subsamples': self.n_sub,
                'eig_time': eig_time,
                'solve_time': solve_time,
                'total_train_time': total_train_time,
                'use_incremental': use_incremental,
            }
            
            # Compute approximation errors if reference is available
            if self._reference_B is not None:
                self.last_metrics['operator_error_hs'] = self.compute_operator_error()
            if self._reference_r is not None:
                self.last_metrics['reward_error_rkhs'] = self.compute_reward_error_rkhs()
                self.last_metrics['reward_error_inf'] = self.compute_reward_error_inf()

    def set_reference_operator(self, B_ref: jnp.ndarray):
        """Set reference operator for error computation."""
        self._reference_B = B_ref
        
    def set_reference_reward(self, r_ref: jnp.ndarray):
        """Set reference reward for error computation."""
        self._reference_r = r_ref
        
    def compute_operator_error(self) -> float:
        """
        Compute Hilbert-Schmidt norm of operator error ||T - T_n||_HS.
        
        Returns:
            Hilbert-Schmidt norm of the difference
        """
        if self._reference_B is None or self.B is None:
            return 0.0
            
        # Handle shape mismatch (reference may have different subsample)
        if self._reference_B.shape != self.B.shape:
            return float('nan')
            
        diff = self._reference_B - self.B
        return float(jnp.sqrt(jnp.sum(diff ** 2)))
    
    def compute_reward_error_rkhs(self) -> float:
        """
        Compute RKHS norm of reward error ||r - r_n||_G.
        
        Returns:
            RKHS norm of reward error (or L2 proxy)
        """
        if self._reference_r is None or self.r is None:
            return 0.0
            
        if self._reference_r.shape != self.r.shape:
            return float('nan')
            
        diff = self._reference_r - self.r
        
        try:
            # Try to compute exact RKHS norm
            if self.K_sub_sub is not None:
                alpha = jnp.linalg.solve(
                    self.K_sub_sub + 1e-6 * jnp.eye(self.K_sub_sub.shape[0]), 
                    diff
                )
                rkhs_norm_sq = float(jnp.dot(alpha.flatten(), diff.flatten()))
                if rkhs_norm_sq >= 0:
                    return float(jnp.sqrt(rkhs_norm_sq))
        except:
            pass
        
        # Fallback to L2 norm
        return float(jnp.linalg.norm(diff))
    
    def compute_reward_error_inf(self) -> float:
        """
        Compute uniform norm of reward error ||r - r_n||_inf.
        
        Returns:
            Maximum absolute difference
        """
        if self._reference_r is None or self.r is None:
            return 0.0
            
        if self._reference_r.shape != self.r.shape:
            return float('nan')
            
        return float(jnp.max(jnp.abs(self._reference_r - self.r)))
    
    def compute_predicted_rewards(self, X_test: jnp.ndarray, A_test: jnp.ndarray) -> jnp.ndarray:
        """
        Compute predicted rewards at test points.
        
        Args:
            X_test: Test states
            A_test: Test actions
            
        Returns:
            Predicted rewards
        """
        if self.r is None or self.X_sub is None:
            return jnp.zeros(X_test.shape[0])
            
        K_test_sub = self.kernel(X_test, self.X_sub) * dirac_kernel(A_test, self.A_sub)
        return K_test_sub @ self.r
    
    def compute_predicted_transitions(self, X_test: jnp.ndarray, A_test: jnp.ndarray) -> jnp.ndarray:
        """
        Compute predicted transition embeddings at test points.
        
        Args:
            X_test: Test states
            A_test: Test actions
            
        Returns:
            Predicted transition embeddings (n_test, n_sub)
        """
        if self.B is None or self.X_sub is None:
            return jnp.zeros((X_test.shape[0], self.n_sub if self.n_sub else 1))
            
        K_test_sub = self.kernel(X_test, self.X_sub) * dirac_kernel(A_test, self.A_sub)
        return K_test_sub @ self.B
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the last computed metrics."""
        return self.last_metrics.copy()

    def __getstate__(self):
        """ Prepare the object for pickling by returning necessary attributes. """
        return {
            'n': self.n,
            'n_sub': self.n_sub,
            'above_threshold_count': self.above_threshold_count,
            'stop_collection': self.stop_collection,
            'A': self.A,
            'X': self.X,
            'Y_transitions': self.Y_transitions,
            'Y_rewards': self.Y_rewards,
            'X_sub': self.X_sub,
            'A_sub': self.A_sub,
            'K_full_sub': self.K_full_sub,
            'K_transitions_sub': self.K_transitions_sub,
            'K_sub_sub': self.K_sub_sub,
            'sub_indices': self.sub_indices,
            'n_sub': self.n_sub,
            'n_components': self.n_components,
            '_SUBSAMPLE_HAS_BEEN_CALLED': self._SUBSAMPLE_HAS_BEEN_CALLED,
            # Woodbury state
            'use_woodbury': self.use_woodbury,
            'M_inv': self.M_inv,
            'L_sub_sqrt': self.L_sub_sqrt,
            '_n_at_last_train': self._n_at_last_train,
            '_pending_K_rows': self._pending_K_rows,
            '_has_trained': self._has_trained,
        }

    def __setstate__(self, state):
        """ Restore the object's state. """
        self.n = state['n']
        self.n_sub = state['n_sub']
        self.above_threshold_count = state['above_threshold_count']
        self.stop_collection = state['stop_collection']
        self.A = state['A']
        self.X = state['X']
        self.Y_transitions = state['Y_transitions']
        self.Y_rewards = state['Y_rewards']
        self.X_sub = state['X_sub']
        self.K_transitions_sub = state['K_transitions_sub']
        self.A_sub = state['A_sub']

        self.K_full_sub = state['K_full_sub']
        self.K_transitions_sub = state['K_transitions_sub']
        self.K_sub_sub = state['K_sub_sub']

        self.sub_indices = state['sub_indices']

        self.n_sub = state['n_sub']
        self.n_components = state['n_components']

        self._SUBSAMPLE_HAS_BEEN_CALLED = state['_SUBSAMPLE_HAS_BEEN_CALLED']

        # Woodbury state (with defaults for backwards compatibility)
        self.use_woodbury = state.get('use_woodbury', True)
        self.M_inv = state.get('M_inv', None)
        self.L_sub_sqrt = state.get('L_sub_sqrt', None)
        self._n_at_last_train = state.get('_n_at_last_train', 0)
        self._pending_K_rows = state.get('_pending_K_rows', None)
        self._has_trained = state.get('_has_trained', False)

