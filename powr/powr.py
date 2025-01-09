import os
import time
import warnings
import argparse
import jax.numpy as jnp
import gymnasium as gym
from powr.utils import *
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings(
    "ignore", category=TqdmExperimentalWarning
)  # Remove experimental warning
from powr.MDPManager import MDPManager

class POWR:
    def __init__(
            self, 
            env, 
            eval_env, 
            args,
            eta=0.1, 
            la=0.1, 
            gamma=0.99, 
            kernel=None,
            n_reduced_rank=1000,
            subsamples=100,
            q_memories=10,
            delete_Q_memory=False,
            early_stopping=None,
            tensorboard_writer=None,
            starting_logging_epoch=0,
            starting_logging_timestep=0,
            run_path=None,
            seed=None,
            checkpoint=None,
            device='cpu',
            offline=False,
            ):
        """
        Initialize the POWR algorithm with the necessary parameters.
        
        Args:
            env (gym.Env): The training environment.
            eval_env (gym.Env): The evaluation environment.
            args (argparse.Namespace): The arguments for the POWR algorithm.
            eta (float): The learning rate.
            la (float): The regularization parameter.
            gamma (float): The discount factor.
            kernel (function): The kernel function to use for the MDP manager.
            n_reduced_rank (int): The number of reduced rank components to use for the kernel approximation.
            subsamples (int): The number of subsamples to use for the kernel approximation.
            q_memories (int): The number of Q-function memories to store.
            delete_Q_memory (bool): Whether to delete the Q-function memory after training.
            tensorboard_writer (SummaryWriter): The TensorBoard writer.
            starting_logging_epoch (int): The starting epoch for logging.
            starting_logging_timestep (int): The starting timestep for logging
            run_path (str): The path to the run directory.
            seed (int): The random seed.
            checkpoint (str): The path to the checkpoint file.
            device (str): The device to run the algorithm on.
        """

        self.env = env
        self.eval_env = eval_env
        assert isinstance(env, gym.experimental.vector.sync_vector_env.SyncVectorEnv), f"env must be created with make_vec_env to obtain type gym.vector.VectorEnv, got {type(env)}"
        assert isinstance(eval_env, gym.experimental.vector.sync_vector_env.SyncVectorEnv), f"env must be created with make_vec_env to obtain type gym.vector.VectorEnv, got {type(eval_env)}"
        self.args = args # TODO cercare di rimuovere, bisogna vedere sul load checkpoint
        self.eta = eta
        self.la = la
        self.gamma = gamma
        assert kernel is not None, "Kernel function must be provided."
        self.subsamples = subsamples
        self.q_memories = q_memories
        self.delete_Q_memory = delete_Q_memory
        self.tensorboard_writer = tensorboard_writer
        self.run_path = run_path
        self.log_file = open(os.path.join((self.run_path), "log_file.txt"), "a", encoding="utf-8")
        self.seed = seed        
        self.checkpoint = checkpoint
        self.device = device
        self.offline = offline

        if early_stopping is not None:
            assert env.spec.reward_threshold is not None, "Environment must have a reward threshold for early stopping."
            assert run_path is not None, "Run path must be provided for early stopping."

        # ** Initialize Logging **
        self.total_timesteps = starting_logging_timestep
        self.starting_epoch = starting_logging_epoch


        # ** MDP manager Settings**
        self.mdp_manager = MDPManager(
            self.env,
            self.eval_env,
            eta=self.eta,
            la=self.la,
            kernel=kernel,
            gamma=self.gamma,
            n_subsamples=self.subsamples,
            early_stopping=early_stopping,
            n_reduced_rank=n_reduced_rank,
            seed=self.seed,
            log_path=self.run_path,
        )
        if self.checkpoint is not None:
            mdp_manager_file = os.path.join(self.checkpoint, "mdp_manager.pkl")
            self.mdp_manager.load_checkpoint(mdp_manager_file)
            print("Loaded from Checkpoint")
            
    
    def train(self, 
              epochs=1000, 
              warmup_episodes=1, 
              train_episodes=1,
              eval_episodes=1,
              iterations_pmd=1,
              eval_every=1,
              save_gif_every=None,
              save_checkpoint_every=None,
              args_to_save=None,

              ):
        """
        Train the POWR algorithm on the given environment.
        
        Args:
            epochs (int): The number of training epochs.
            warmup_episodes (int): The number of warmup episodes.
            training_episodes (int): The number of episodes to use for training.
            eval_episodes (int): The number of episodes to use for evaling.
            iterations_pmd (int): The number of PMD iterations.
            eval_every (int): Evaluate the policy every n epochs.
            save_gif_every (int): Save a GIF every n epochs.
            save_checkpoint_every (int): Save a checkpoint every n epochs.
            args_to_save (argparse.Namespace): The arguments to save in the checkpoint.
        """

        assert  warmup_episodes > 0, "Number of warmup episodes must be greater than 0"

        if save_checkpoint_every is not None:
            assert self.run_path is not None, "Run path must be provided to save checkpoints."

        #** Warmup the models **
        if warmup_episodes > 0: 
            start_sampling = time.time()
            train_result, timesteps = self.mdp_manager.collect_data(warmup_episodes)
            t_sampling = time.time() - start_sampling

        self.total_timesteps += timesteps


        for i in tqdm(range(epochs)):

            # ** Training the models with previously collected data**
            start_training = time.time()
            self.mdp_manager.train()
            t_training = time.time() - start_training

            # ** Applying Policy Mirror Descent to Policy**
            start_pmd = time.time()
            self.mdp_manager.policy_mirror_descent(iterations_pmd)
            t_pmd = time.time() - start_pmd

            # ** Evaluate the policy every <eval_every> epochs **
            if i%eval_every == 0:
                if eval_episodes > 0:
                    start_eval = time.time()
                    eval_result = self.mdp_manager.eval(
                        eval_episodes
                    )
                    t_eval = time.time() - start_eval
                else:
                    t_eval = None
                    eval_result = None
            else:
                    t_eval = None
                    eval_result = None


            # ** Save gif ** # TODO evaling
            if save_gif_every is not None and i % save_gif_every == 0:
                self.mdp_manager.eval(
                    1,
                    plot=True,
                    wandb_log=(self.offline == False),
                    path=self.run_path,
                    current_epoch=i + self.starting_epoch,
                )
            
            execution_time = time.time() - start_sampling

            # ** Log data **
            log_epoch_statistics(
                self.tensorboard_writer,
                self.log_file,
                i + self.starting_epoch,
                eval_result,
                train_result,
                train_episodes,
                iterations_pmd,
                warmup_episodes,
                self.total_timesteps,
                t_sampling,
                t_training,
                t_pmd,
                t_eval,
                execution_time,
            )

            
            # Save checkpoint every <save_checkpoint_every> epochs 
            if save_checkpoint_every is not None and i % save_checkpoint_every == 0 and i > 0:
                save_checkpoint(self.run_path , self.args, self.total_timesteps, i + self.starting_epoch, self.mdp_manager)
                print(f"\033[92mSaved checkpoint at epoch {i + self.starting_epoch} {RESET}")

            # No need to store new data if it's the last epoch
            if i == epochs - 1:
                break
            
            # ** Collect data for the next epoch **
            start_sampling = time.time()
            train_result, timesteps = self.mdp_manager.collect_data(
                train_episodes
            )
            t_sampling = time.time() - start_sampling

            self.total_timesteps += timesteps

            self.mdp_manager.reset_Q(q_memories = self.q_memories)

            if self.delete_Q_memory:
                self.mdp_manager.delete_Q_memory()
            
        save_checkpoint(self.run_path , args_to_save, self.total_timesteps, i + self.starting_epoch if self.checkpoint is not None else i, self.mdp_manager)
    

    
    def evaluate(self, episodes):
        """
        Evaluate the policy on the environment without updating it.
        
        Args:
            episodes (int): Number of episodes to evaluate.
        
        Returns:
            avg_reward (float): The average reward over the evaluation episodes.
        """

        return self.mdp_manager.eval(episodes)
        