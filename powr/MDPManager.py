
import os
import cv2
import jax
import time
import wandb
import pickle
import logging
import imageio
import numpy as np
import jax.numpy as jnp
import gymnasium as gym

from powr.Qmodel import Qmodel
from powr.kernels import softmax
from powr.kernels import dirac_kernel
from powr.IncrementalRLS import IncrementalRLS

class MDPManager:

    def __init__(
        self,
        env,
        eval_env,
        gamma=0.95,
        eta=1,
        la=1e-3,
        kernel=None,
        n_subsamples=None,
        n_reduced_rank=1000,
        early_stopping=None,    
        eps_softmax=1e-9,
        seed=None,
        log_path=None,

    ):

        assert kernel is not None
        assert n_subsamples is not None

        self.kernel = kernel
        self.n_subsamples = n_subsamples
       

        # ** Environment Settings **
        self.env = env
        self.eval_env = eval_env
        # Set the seed
        if seed is not None:
            self.env.reset(seed=seed)
            self.eval_env.reset(seed=seed)
        self.seed = seed
        # Number of actions
        self.n_actions =  self.env.single_action_space.n

        # ** Hyperparameters **
        self.gamma = gamma
        self.eta = eta
        self.la = la
        self.eps_softmax = eps_softmax
        self.early_stopping = early_stopping
        self.n_reduced_rank = n_reduced_rank


        # to keep track of the training expoenents
        self.cum_train_exponent = None

        # previous  models for the cumulative Q functions
        self.f_prev_cumQ_models = []
        self.f_prev_exponents = None

        self.action_one_hot = jnp.eye(self.n_actions)

        self.softmax = softmax()

        self.f_cumQ_weights = None
        self.f_Q_mask = None

        self.log_path = log_path    

        self.FTL = IncrementalRLS(
            kernel=self.kernel,
            n_actions=self.n_actions,
            la=self.la,
            n_subsamples=self.n_subsamples,
            early_stopping=self.early_stopping,
            log_path=self.log_path,
            n_components=self.n_reduced_rank,
        )

    def check_data_collected_but_not_trained(self):
        assert not self._DATA_COLLECTED_BUT_NOT_TRAINED

    # Update the Q function
    def update_Q(self):

        self.check_data_collected_but_not_trained()

        assert self.f_cumQ_weights is not None

        V = self.FTL.V

        start_exponents = time.time()
        f_exponents = (
            self.f_prev_exponents
            + self.eta * self.FTL.K_transitions_sub @ self.f_cumQ_weights
        )
        print(f"exponent computation time: {time.time() - start_exponents}")
        f_pi = self.softmax(f_exponents)

        pPit = self.FTL.K_transitions_sub.reshape(
            self.FTL.n, self.FTL.n_sub, 1
        ) * f_pi.reshape(self.FTL.n, 1, self.n_actions) 
        pPit = pPit[:, jnp.arange(self.FTL.n_sub), self.FTL.A_sub] # M matrix in paper

        start_bing_M = time.time()
        f_big_M = jnp.eye(self.FTL.n_sub) - (self.gamma) * self.FTL.B @ pPit # Id - gamma * B * M
        print(f"big M computation time: {time.time() - start_bing_M}")

        start_tmp_q = time.time()
        f_tmp_Q = V.T @ jnp.linalg.solve(V @ f_big_M @ V.T, V @ self.FTL.r) # c
        print(f"tmp Q computation time: {time.time() - start_tmp_q}")

        self.f_cumQ_weights += f_tmp_Q * self.f_Q_mask
        

    # def get_Q_pi(self, states, actions):
    #     V = self.FTL.V

    #     f_exponents = (
    #         self.f_prev_exponents
    #         + self.eta * self.FTL.K_transitions_sub @ self.f_cumQ_weights
    #     )
    #     f_pi = self.softmax(f_exponents)

    #     pPit = self.FTL.K_transitions_sub.reshape(
    #         self.FTL.n, self.FTL.n_sub, 1
    #     ) * f_pi.reshape(self.FTL.n, 1, self.n_actions) 
    #     pPit = pPit[:, jnp.arange(self.FTL.n_sub), self.FTL.A_sub] # M matrix in paper
    #     f_big_M = jnp.eye(self.FTL.n_sub) - (self.gamma) * self.FTL.B @ pPit # Id - gamma * B * M
    #     f_tmp_Q = V.T @ jnp.linalg.solve(V @ f_big_M @ V.T, V @ self.FTL.r) # c

    #     # print("kernel",jnp.sum(self.FTL.kernel(states, self.FTL.X_sub), )


    # Delete the Q function from memory
    def delete_Q_memory(self):
        self.f_prev_cumQ_models = []
        self.f_prev_exponents = None

    # Reset the Q function
    def reset_Q(self, q_memories  = None):

        # Append the current Q function to the list of functions
        self.f_prev_cumQ_models.append(
            Qmodel(
                kernel=self.kernel,
                Q=self.eta * self.f_cumQ_weights,
                X_sub=self.FTL.X_sub,
            )
        )

        if q_memories is not None and q_memories < len(self.f_prev_cumQ_models):
            self.f_prev_cumQ_models = self.f_prev_cumQ_models[-q_memories :]

        self.FTL.reset()

        self.f_cumQ_weights = None

    def subsample(self):

        self.FTL.subsample()

    def evaluate_pi(self, state):
        if self.f_cumQ_weights is None:
            return jnp.ones(self.n_actions) / self.n_actions

        jnp_state = jnp.array(state).reshape(1, -1)
        
        exponent = (
            self.eta
            * self.FTL.kernel(jnp.array(state).reshape(1, -1), self.FTL.X_sub)
            @ self.f_cumQ_weights
        )

        for model in self.f_prev_cumQ_models:
            exponent += model.evaluate(jnp_state)

        pi = self.softmax(exponent)
 
        return pi
    
    def sample_action(self, states):

        # Parallel policy evaluation
        probs = jax.vmap(self.evaluate_pi)(states)

        actions = []
        for p in probs:

            if jnp.isnan(p).any():
                raise ValueError("Error: NaN in the results of the training")
            
            p = jnp.asarray(p).astype("float64")
            p = p.squeeze()
            p = p / p.sum()

            try:
                action = np.random.choice(self.n_actions, p=p)
            except:
                raise ValueError("Error: NaN in the results of the training")
  
            actions.append(action)

        return jnp.array(actions), probs
    
    def collect_data(self, n_episodes=1):

        total_timesteps = 0
        f_X, f_Y_transitions, f_Y_rewards, f_A = None, None, None, None

        cum_rewards = np.zeros((n_episodes, self.env.num_envs))
        for episode_id in range(n_episodes):
            done = jnp.zeros(self.env.num_envs, dtype=bool)
            old_terminated = jnp.zeros(self.env.num_envs, dtype=bool)
            old_truncated = jnp.zeros(self.env.num_envs, dtype=bool)
    
            # Vectorized reset of environments
            states, _ = self.env.reset()    
           
            start_sampling_episode = time.time()
            while True:
                actions, pi = self.sample_action(states)
                # round all elements in pi to the third decimal
                pi = jnp.round(pi, 3)

                total_timesteps += jnp.sum(~done)
                
                # Perform vectorized step with all environments
                new_states, rewards, terminations, truncations, infos = self.env.step(np.array(actions))  # Convert actions to CPU
                # if a reward inside rewards array grather than zero print rewards
                if jnp.any(rewards > 0):
                    print(rewards)

                # Update cumulative rewards
                cum_rewards[episode_id] += jnp.multiply(rewards,~done)

                truncations = jnp.logical_or(truncations, old_truncated)
                terminations = jnp.logical_or(terminations, old_terminated)
       

                # update the next_states correctly when the episod is truncated
                new_truncation_mask = jnp.logical_xor(old_truncated, truncations)
                assert jnp.sum(old_truncated) <= jnp.sum(truncations)

                if new_truncation_mask.any():

                    # Convert each individual array in final_obs to a JAX array
                    final_obs = jnp.stack([jnp.array(obs) for obs in infos['final_observation'][new_truncation_mask]])

                    # Now assign final_obs to the new_states using the mask
                    new_states[new_truncation_mask] = final_obs

                # the new truncated envs -> new environments that finished succesfully
                new_terminated_mask = jnp.logical_xor(old_terminated, terminations)
                # if the episode terminated, record the last state as a sink state
                if new_terminated_mask.any():

                    # Convert each individual array in final_obs to a JAX array
                    final_obs = jnp.stack([jnp.array(obs) for obs in infos['final_observation'][new_terminated_mask]])

                    # Now assign final_obs to the new_states using the mask
                    new_states[new_terminated_mask] = final_obs

                    # TODO: check if it is necessary or duplicated
                    # f_X = jnp.vstack((f_X, states[new_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else states[new_terminated_mask]))
                    # f_Y_transitions = jnp.vstack((f_Y_transitions, new_states[new_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else new_states[new_terminated_mask]))
                    # f_Y_rewards = jnp.hstack((f_Y_rewards, rewards[new_terminated_mask]))
                    # f_A = jnp.hstack((f_A, actions[new_terminated_mask]))

                    # if the episode terminated, record the last state as a sink state #TODO
                    rewards = jnp.where(new_terminated_mask, 0, rewards)

                    for a in range(self.n_actions):
                        actions = jnp.where(new_terminated_mask, a, actions)
                        f_X = jnp.vstack((f_X, new_states[new_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else new_states[new_terminated_mask]))
                        f_Y_transitions = jnp.vstack((f_Y_transitions, new_states[new_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else new_states[new_terminated_mask]))
                        f_Y_rewards = jnp.hstack((f_Y_rewards, rewards[new_terminated_mask]))
                        f_A = jnp.hstack((f_A, actions[new_terminated_mask]))


                non_terminated_mask = ~done
                
                # Collect data only for non-terminated environments
                if f_X is None:  # First iteration
                    f_X = states[non_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else states[non_terminated_mask]
                    f_Y_transitions = new_states[non_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else new_states[non_terminated_mask]
                    f_Y_rewards = rewards[non_terminated_mask] 
                    f_A = actions[non_terminated_mask]
                else:  # After first iteration, concatenate
                    f_X = jnp.vstack((f_X, states[non_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else states[non_terminated_mask]))
                    f_Y_transitions = jnp.vstack((f_Y_transitions, new_states[non_terminated_mask].reshape(-1, 1) if isinstance(self.env.single_observation_space, gym.spaces.Discrete) else new_states[non_terminated_mask]))
                    f_Y_rewards = jnp.hstack((f_Y_rewards, rewards[non_terminated_mask]))
                    f_A = jnp.hstack((f_A, actions[non_terminated_mask]))
                
                # Update states to the new states 
                states = new_states 
                old_truncated = truncations
                old_terminated = terminations          

                # Update done environments
                done = jnp.logical_or(terminations, truncations)
                if done.all():
                    break
        
            print("episode_id", episode_id, "sampled in", time.time() - start_sampling_episode)
        self.FTL.collect_data(f_A, f_X, f_Y_transitions, f_Y_rewards.reshape(-1, 1), cum_rewards.flatten().mean() >= self.env.spec.reward_threshold, seed = self.seed)

        self.last_f_X = f_X
        self.last_f_A = f_A

        self._DATA_COLLECTED_BUT_NOT_TRAINED = True

        return cum_rewards.flatten().mean(), total_timesteps

    def eval(self, n_episodes=1, plot=False, path=None, wandb_log=False, current_epoch=None):

        cum_rewards = np.zeros((n_episodes, self.eval_env.num_envs))
        total_timesteps = 0

        for episode_id in range(n_episodes):
            done = jnp.zeros(self.eval_env.num_envs, dtype=bool)
            old_terminated = jnp.zeros(self.eval_env.num_envs, dtype=bool)
            old_truncated = jnp.zeros(self.eval_env.num_envs, dtype=bool)
    
            # Vectorized reset of environments
            states, _ = self.eval_env.reset()     
            images = []
           
            while True:
                actions, pi = self.sample_action(states)
                total_timesteps += jnp.sum(~done)
                
                # Perform vectorized step with all environments
                new_states, rewards, terminations, truncations, infos = self.eval_env.step(np.array(actions))  # Convert actions to CPU
            
                # Update cumulative rewards
                cum_rewards[episode_id] += jnp.multiply(rewards,~done)

                truncations = jnp.logical_or(truncations, old_truncated)
                terminations = jnp.logical_or(terminations, old_terminated)
                
                # Update states to the new states 
                states = new_states 
                old_truncated = truncations
                old_terminated = terminations

                if plot:
                    img =self.eval_env.envs[0].render()
    
                    # Ensure the image is a NumPy array
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    # write the action on the right left corner of the image (in green) font 1 and thickness 2
                    img = cv2.putText(
                        img,
                        f"Action: {actions[0]} - Pi: {pi[0]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    images.append(img)

                # Update done environments
                done = jnp.logical_or(terminations, truncations)
                if done.all():
                    break

            # save the gif
            if plot:
                gif_name = f"epoch={current_epoch}-reward-{cum_rewards[episode_id][0]}.gif"
                if path is None:
                    path = "./gifs/tmp"
                    if os.path.isdir(path) is False:
                        os.mkdir(path)
                    imageio.mimsave(f"{path}/{gif_name}", images)
                else:
                    imageio.mimsave(f"{path}/{gif_name}", images)

                if wandb_log:

                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "video": wandb.Video(path + "/" + gif_name),
                        }
                    )
                break

        return cum_rewards.flatten().mean()

    def train(self):

        self.FTL.train()

        if self.f_cumQ_weights is None and self.FTL.n_sub > 0:
            self.f_cumQ_weights = jnp.zeros((self.FTL.n_sub, self.n_actions))
            self.f_Q_mask = self.action_one_hot[self.FTL.A_sub]

        self.f_prev_exponents = jnp.zeros((self.FTL.n, self.n_actions))
        for model in self.f_prev_cumQ_models:
            self.f_prev_exponents += model.evaluate(self.FTL.Y_transitions)

        self._DATA_COLLECTED_BUT_NOT_TRAINED = False

    def policy_mirror_descent(self, n_iter):

        for _ in range(n_iter):
            self.update_Q()

    def save_checkpoint(self, filename):
        """ Save the important state of the instance as a checkpoint. """
        with open(filename, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

  
    def load_checkpoint(self, filename):
        """ Load the important state of the instance from a checkpoint. """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
            self.__setstate__(state)
            return self

    def __getstate__(self):
        """ Prepare the object for pickling, returning only the relevant state. """
        state = {
            'cum_train_exponent': self.cum_train_exponent,
            'f_prev_cumQ_models': [model.__getstate__() for model in self.f_prev_cumQ_models],
            'f_prev_exponents': self.f_prev_exponents,
            'f_cumQ_weights': self.f_cumQ_weights,
            'f_Q_mask': self.f_Q_mask,
            'FTL': self.FTL.__getstate__()  # Save the state of the IncrementalRLS instance
        }
        return state

    def __setstate__(self, state):
        """ Restore the object's state. """
        self.cum_train_exponent = state.get('cum_train_exponent', None)
        
        # Recreate the FastQmodel instances from the pickled state
        self.f_prev_cumQ_models = [Qmodel(kernel=self.kernel, **model_state) for model_state in state.get('f_prev_cumQ_models', [])]
        
        self.f_prev_exponents = state.get('f_prev_exponents', None)
        self.f_cumQ_weights = state.get('f_cumQ_weights', None)
        self.f_Q_mask = state.get('f_Q_mask', None)

        # Restore the IncrementalRLS instance state
        self.FTL = IncrementalRLS(kernel=self.kernel, n_actions=self.n_actions, la=self.la, n_subsamples=self.n_subsamples, early_stopping=self.early_stopping, log_path=self.log_path)
        self.FTL.__setstate__(state['FTL'])

