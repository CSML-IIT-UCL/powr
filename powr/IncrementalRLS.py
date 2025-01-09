import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import logging

from powr.kernels import dirac_kernel

class IncrementalRLS:
    def __init__(self, kernel=None, n_actions=None, la=1e-3, n_subsamples=None, early_stopping=None, log_path=None, n_components=1000):

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
        self.n_components = n_components

        self.early_stopping_episodes= early_stopping # TODO: find a new effective strategy to early stop the collection
        self.log_path = log_path
        self.above_threshold_count = 0
        self.stop_collection = False

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
        self.K_full_sub = jnp.vstack(
            [self.K_full_sub, Knew[: X.shape[0]] * dirac_kernel(A, self.A_sub)]
        )
        self.K_transitions_sub = jnp.vstack(
            [self.K_transitions_sub, Knew[X.shape[0] :]]
        )


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

        self._SUBSAMPLE_HAS_BEEN_CALLED = True

    def train(self):

        if not self._SUBSAMPLE_HAS_BEEN_CALLED:
            self.subsample()

        V, W = jax.lax.linalg.eigh(self.K_sub_sub)
        effective_components = min(self.K_sub_sub.shape[0], self.n_components)
        self.V = V[:, -effective_components : ].T

        L = jax.lax.linalg.cholesky(
            self.K_full_sub.T @ self.K_full_sub
            + self.n * self.la * self.K_sub_sub
            + 1e-6 * jnp.eye(self.K_full_sub.shape[1])
        )

        if jnp.isnan(L).any():
            raise ValueError("Error: NaN in the results of the training for Chol")

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

        logging.debug(f"Results: {W}")
        self.r = W[:, -1].reshape(-1, 1)
        # logging.debug(f"Results: {self.r}\n\n\n")
        self.B = W[:, :-1]

        # check if the results contain nan
        if jnp.isnan(self.r).any() or jnp.isnan(self.B).any():
            raise ValueError("Error: NaN in the results of the training")

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
            '_SUBSAMPLE_HAS_BEEN_CALLED': self._SUBSAMPLE_HAS_BEEN_CALLED
            

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

