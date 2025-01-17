import os
import jax
import wandb
import socket
import logging
import warnings
import argparse
import gymnasium as gym
from pprint import pprint
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import TqdmExperimentalWarning

jax.config.update("jax_enable_x64", True)
os.environ["WANDB_START_METHOD"] = "thread"
warnings.filterwarnings(
    "ignore", category=TqdmExperimentalWarning
)  # Remove experimental warning

from powr.utils import *
from powr.wrappers import *
from powr.powr import POWR
from powr.kernels import dirac_kernel, gaussian_kernel, gaussian_kernel_diag

logging.basicConfig(level=logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('tensorboardX').setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Taxi-v3", type=str, help="Train gym env [LunarLander-v2, MountainCar-v0, CartPole-v1, Pendulum-v1]",)
    parser.add_argument("--group", default=None, type=str, help="Wandb run group")
    parser.add_argument("--project", default=None, type=str, help="Wandb project")
    parser.add_argument("--la", default=1e-13, type=float, help="Regularization for the action-value function estimators",)
    parser.add_argument("--eta", default=1, type=float, help="Step size of the Policy Mirror Descent")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--sigma", default=0.2, type=float, help="Std of Gaussian Kernel")
    parser.add_argument("--q-mem", "-qm", default=1, type=int, help="Number of Q-memories to use to use, i.e., batch size for Q functions",)
    parser.add_argument("--delete-Q-memory", "-dqm", default=False, action="store_true", help="Delete the previously estimated Q functions",)
    parser.add_argument("--early-stopping", "-es", default=None, type=int, help="Number of consecutive episodes above <env> reward threshold for early stopping the data collection",)
    parser.add_argument("--warmup-episodes", "-we", default=20, type=int, help="Number of warmups epochs for initializing the P i.e. (transition probability) and Q matrices",)
    parser.add_argument("--epochs", "-e", default=50, type=int, help="Number of training epochs, i.e. Data Sampling, P computation, Policy Mirror Descent, and Testing",)
    parser.add_argument("--train-episodes","-te", default=10, type=int, help="Number of episodes used to sample for each epoch",)
    parser.add_argument("--parallel-envs", "-pe", default=10, type=int, help="Number of parallel environments",)
    parser.add_argument("--subsamples", "-subs", default=2000, type=int, help="Number of subsamples for nystrom kernel",)
    parser.add_argument("--n_reduced_rank", "-nrr", default=5000, type=int, help="Number of ...",)
    parser.add_argument("--iter-pmd", "-pmd", default=10, type=int, help="Number of iteration to update policy parameters in an off-policy manner", )
    parser.add_argument("--eval-episodes", "-ee", default=1, type=int, help="Number of evaluation episodes")
    parser.add_argument("--save-gif-every","-sge", default=5, type=int, help="Save gif every <save-gif-every> epochs",)
    parser.add_argument("--save-checkpoint-every","-sce", default=20, type=int, help="Save checkpoint every <save-checkpoint-every> epochs",)
    parser.add_argument("--eval-every", default=10, type=int, help="Evaluate policy every <eval-every> epochs",)
    parser.add_argument("--seed", default=6, type=int, help="seed")
    parser.add_argument("--checkpoint", "-c", default=None, type=str, help="Checkpoint path, None means no checkpoint loading",)
    parser.add_argument("--device", type=str, default="cpu",  help="Device setting <cpu> or <gpu>",)
    parser.add_argument("--notes", default=None, type=str, help="Wandb notes")
    parser.add_argument("--tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: --tags 'optimized' 'baseline' ",)
    parser.add_argument("--offline", default=True, action="store_true", help="Offline run without wandb",)
    args = parser.parse_args()
    args.algo = "powr"

    return args


def parse_env(env_name, parallel_envs, sigma):
    if env_name == "Taxi-v3":
        env = gym.make_vec("Taxi-v3",  num_envs=parallel_envs, vectorization_mode="sync", render_mode="rgb_array")
        kernel = dirac_kernel

    elif env_name == "FrozenLake-v1":
        env = gym.make_vec(
            "FrozenLake-v1",  num_envs=parallel_envs, vectorization_mode="sync",
            desc=None,
            map_name="4x4",
            is_slippery=False,
            render_mode="rgb_array",
        )
        kernel = dirac_kernel

    elif env_name == "LunarLander-v2":
        env = gym.make_vec("LunarLander-v2", num_envs=parallel_envs, vectorization_mode="sync", render_mode="rgb_array")
        sigma_ll = [sigma for _ in range(6)]
        sigma_ll += [0.0001, 0.0001]
        kernel = gaussian_kernel_diag(sigma_ll)

    elif env_name == "MountainCar-v0":

        env = gym.make_vec("MountainCar-v0", num_envs=parallel_envs, vectorization_mode="sync", render_mode="rgb_array")
        sigma_mc = [0.1, 0.01]
        kernel = gaussian_kernel_diag(sigma_mc)

    elif env_name == "CartPole-v1":
        env = gym.make_vec("CartPole-v1", num_envs=parallel_envs, vectorization_mode="sync", render_mode="rgb_array")
        kernel = gaussian_kernel(sigma)

    elif env_name == "Pendulum-v1":
        env = gym.make_vec("Pendulum-v1", g=9.81, num_envs=parallel_envs, vectorization_mode="sync", render_mode="rgb_array")
        kernel = gaussian_kernel(sigma)

    else:
        raise ValueError(f"Unknown environment: {args.env}")

    return env, kernel


def get_run_name(args, current_date=None):
    if current_date is None:
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        str(current_date)
        + "_"
        + str(args.env)
        + "_"
        + args.algo
        + "_eta="
        + str(args.eta)
        + "_la="
        + str(args.la)
        + "_train_eps="
        + str(args.train_episodes)
        + "_pmd_iters="
        + str(args.iter_pmd)
        + "_earlystop="
        + str(args.early_stopping)
        + "_seed"
        + str(args.seed)
        + "_"
        + socket.gethostname()
    )


if __name__ == "__main__":
    # ** Run Settings **
    # Parse arguments
    args = parse_args()
    checkpoint = args.checkpoint

    # ** Wandb Settings **
    # Resume Wandb run if checkpoint is provided
    if checkpoint is not None:
        checkpoint_data = load_checkpoint(checkpoint)
        project = args.project

        # Load saved `args`, `total_timesteps`, and `wandb_run_id`
        args = argparse.Namespace(**checkpoint_data["args"])
        total_timesteps = checkpoint_data["total_timesteps"]
        starting_epoch = checkpoint_data["epoch"]
        wandb_run_id = checkpoint_data["wandb_run_id"]
        print("Resuming WandB run: ", wandb_run_id)
        # Resume Wandb run with saved run ID
        wandb.init(
            project=project,
            id=wandb_run_id,  # Use saved Wandb run ID to resume the run
            save_code=True,
            sync_tensorboard=True,
            monitor_gym=True,
            resume="must",
            mode=("online" if not args.offline else "disabled"),
        )

        run_path = f"{checkpoint}/"
    else:
        pprint(vars(args))
        random_string = get_random_string(5)
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        run_path = (
            "runs/"
            + str(args.env)
            + "/"
            + args.algo
            + "/"
            + get_run_name(args, current_date)
            + "_"
            + random_string
            + "/"
        )
        create_dirs(run_path)
        save_config(vars(args), run_path)

        # Initialize wandb
        wandb.init(
            config=vars(args),
            project=("powr" if args.project is None else args.project),
            group=(f"{args.env}/{args.algo}" if args.group is None else args.group),
            name=str(current_date)
            + "_"
            + str(args.env)
            + "_"
            + args.algo
            + "_eta="
            + str(args.eta)
            + "_la="
            + str(args.la)
            + "_train_eps="
            + str(args.train_episodes)
            + "_pmd_iters="
            + str(args.iter_pmd)
            + "_earlystop="
            + str(args.early_stopping)
            + "_seed"
            + str(args.seed)
            + "_"
            + random_string,
            save_code=True,
            sync_tensorboard=True,
            tags=args.tags,
            monitor_gym=True,
            notes=args.notes,
            mode=("online" if not args.offline else "disabled"),
        )
        starting_epoch = 0
        total_timesteps = 0

    # ** Device Settings **
    device_setting = args.device
    if device_setting == "gpu":
        device = jax.devices("gpu")[0]
        jax.config.update("jax_default_device", device)  # Update the default device to GPU

        print(f"Currently running on \033[92mGPU {RESET}")
    elif device_setting == "cpu":
        
        try:
            os.environ["JAX_PLATFORMS"] = "cpu"
            device = jax.devices("cpu")[0]  
            jax.config.update("jax_default_device", device)  # Update the default device to CPU
        except:
            os.environ["JAX_PLATFORMS"] = "cpu"
            jax.config.update("jax_default_device", jax.devices("cpu")[0])

        print(f"Currently running on \033[92mCPU {RESET}")
    else:
        raise ValueError(f"Unknown device setting {device_setting}, please use <cpu> or <gpu>")
    

    # ** Logging Settings **
    # Create tensorboard writer
    writer = SummaryWriter(f"{run_path}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create log file
    log_file = open(os.path.join((run_path), "log_file.txt"), "a", encoding="utf-8")

    # ** Hyperparameters Settings **
    subsamples = args.subsamples
    la = args.la
    eta = args.eta
    gamma = args.gamma
    q_memories = args.q_mem

    parallel_envs = args.parallel_envs
    warmup_episodes = args.warmup_episodes
    assert warmup_episodes > 0, "Number of warmup episodes must be greater than 0"
    if warmup_episodes % parallel_envs != 0:

        warnings.warn(
                f"Number of warmup episodes {warmup_episodes} not divisible by parallel environments {parallel_envs}, considering {(warmup_episodes // parallel_envs + 1)*parallel_envs} warmup episodes",
                UserWarning,
            )        
        warmup_episodes = warmup_episodes//parallel_envs + 1
    else:
        warmup_episodes = warmup_episodes//parallel_envs

    epochs = args.epochs
    train_episodes = args.train_episodes
    if train_episodes % parallel_envs != 0:

        warnings.warn(
                f"Number of training episodes {train_episodes} not divisible by parallel environments {parallel_envs}, considering {(train_episodes // parallel_envs + 1)*parallel_envs} training episodes",
                UserWarning,
            )        
        train_episodes = train_episodes//parallel_envs + 1
    else:
        train_episodes = train_episodes//parallel_envs
    
    iter_pmd = args.iter_pmd
    eval_episodes = args.eval_episodes
    if eval_episodes % parallel_envs != 0:

        warnings.warn(
                f"Number of evaluation episodes {eval_episodes} not divisible by parallel environments {parallel_envs}, considering {(eval_episodes // parallel_envs + 1)*parallel_envs} evaluation episodes",
                UserWarning,
            )        
        eval_episodes = eval_episodes//parallel_envs + 1
    else:
        eval_episodes = eval_episodes//parallel_envs

    assert args.early_stopping is None or args.early_stopping > 0, "Number of early stopping episodes must be greater than 0"
    early_stopping = args.early_stopping/parallel_envs if args.early_stopping is not None else None

    save_gif_every = args.save_gif_every
    eval_every = args.eval_every
    save_checkpoint_every = args.save_checkpoint_every  
    delete_Q_memory = args.delete_Q_memory

    # ** Environment Settings **
    env, kernel = parse_env(args.env, parallel_envs, args.sigma)

    # ** Kernel Settings **
    def to_be_jit_kernel(X, Y):
        return kernel(X, Y)

    jit_kernel = jax.jit(to_be_jit_kernel)
    v_jit_kernel = jax.vmap(jit_kernel) # TODO Not used

    # ** Seed Settings**
    set_seed(args.seed)

    # ** POWR Initialization **
    powr = POWR(
            env, 
            env, 
            args,
            eta=eta, 
            la=la, 
            gamma=gamma, 
            kernel=jit_kernel,
            subsamples=subsamples,
            n_reduced_rank=args.n_reduced_rank,
            q_memories=q_memories,
            delete_Q_memory=delete_Q_memory,
            early_stopping=early_stopping,
            tensorboard_writer=writer,
            starting_logging_epoch=starting_epoch,
            starting_logging_timestep=total_timesteps,
            run_path=run_path,
            seed=args.seed,
            checkpoint=checkpoint,
            device=device_setting,
            offline=args.offline,
        
    )

    # ** Training **
    print(f"\033[1m\033[94mTraining the policy{RESET}")
    powr.train( 
        epochs=epochs,
        warmup_episodes = warmup_episodes,
        train_episodes = train_episodes,
        eval_episodes = eval_episodes,
        iterations_pmd= iter_pmd,
        eval_every=eval_every,
        save_gif_every=save_gif_every,
        save_checkpoint_every=save_checkpoint_every,
        args_to_save=args,
    ) 

    # ** Testing **
    print(f"\033[1m\033[94mTesting the policy{RESET}")
    n_test_episodes = 10
    mean_reward = powr.evaluate(n_test_episodes)

    print(f"Policy mean reward over {n_test_episodes} episodes: {mean_reward}")