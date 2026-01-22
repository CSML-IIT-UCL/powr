import json
import os
import random
import socket
import string
import warnings
from datetime import datetime
from tabulate import tabulate
import wandb
import pickle

import jax
import numpy as np

def save_checkpoint(run_name, args, total_timesteps, epoch, mdp_manager):
    checkpoint_data = {
        "args": vars(args),  # Save arguments
        "total_timesteps": total_timesteps,  # Save total timesteps
        "epoch": epoch,  # Save epoch
        "wandb_run_id": wandb.run.id,  # Save Wandb run ID
        "run_name": run_name,  # Save run name for consistent logging
    }
    
    # Save checkpoint data
    checkpoint_file = os.path.join(run_name, "checkpoint.pkl")
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    # Save the entire mdp_manager object separately
    mdp_manager_file = os.path.join(run_name, "mdp_manager.pkl")
    mdp_manager.save_checkpoint(mdp_manager_file)
    
    
    print(f"Checkpoint and mdp_manager saved at {run_name}")



def load_checkpoint(run_name):
    # Load checkpoint data
    checkpoint_file = os.path.join(run_name, "checkpoint.pkl")
    with open(checkpoint_file, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    
    print(f"Checkpoint and mdp_manager loaded from {run_name}")
    
    return checkpoint_data


def get_run_name(args, current_date=None):
    if current_date is None:
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        str(current_date)
        + "_"
        + str(args.env)
        + "_"
        + str(args.algo)
        + "_t"
        + str(args.timesteps)
        + "_HiddenL"
        + str(args.hidden_layers)
        + (f"_activation-{args.activation}" if args.activation is not None else "")
        + "_seed"
        + str(args.seed)
        + "_"
        + socket.gethostname()
    )


def get_random_string(n=5):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


def set_seed(seed):
    # Seeding every module
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    jax.random.key(seed)


def create_dir(path):
    try:
        os.mkdir(os.path.join(path))
    except OSError as error:
        # print('Dir already exists')
        pass


def create_dirs(path):
    try:
        os.makedirs(os.path.join(path))
    except OSError as error:
        pass


def save_config(config, path):
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file)
    return

# ANSI escape codes for yellow
YELLOW = "\033[93m"  # Yellow text
RESET = "\033[0m"    # Reset to default color


# Custom warning handler
def custom_warning_handler(message, category, *args, **kwargs):
    # Print the warning message in yellow
    print(f"{YELLOW}{category.__name__}: {message}{RESET}")

# Set the custom warning handler
warnings.showwarning = custom_warning_handler

def log_epoch_statistics(writer, log_file, epoch, eval_result, train_result, n_train_episodes,
                         n_iter_pmd, n_warmup_episodes, total_timesteps, 
                         t_sampling, t_training, t_pmd, t_eval, execution_time,
                         extra_metrics=None
                         ):
    # Log to Tensorboard
    global_step = epoch

    if eval_result is not None:
        writer.add_scalar("eval reward", eval_result, global_step)
    writer.add_scalar("train reward", train_result, global_step)
    writer.add_scalar(
        "Sampling and Updating steps",
        n_warmup_episodes + epoch * (n_train_episodes + n_iter_pmd),
        global_step,
    )
    writer.add_scalar("Epoch", epoch, global_step)
    writer.add_scalar("Train Episodes", n_warmup_episodes + epoch * n_train_episodes, global_step)
    writer.add_scalar("timestep", total_timesteps, global_step)
    writer.add_scalar("Epoch and Warmup ", epoch + n_warmup_episodes, global_step)
    writer.add_scalar("Sampling Time", t_sampling, global_step)
    writer.add_scalar("Training Time", t_training, global_step)
    writer.add_scalar("PMD Time", t_pmd, global_step)
    if t_eval is not None:
        writer.add_scalar("Eval Time", t_eval, global_step)
    writer.add_scalar("Execution Time", execution_time, global_step)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Train reward", fancy_float(train_result)],
    ])
    
    if eval_result is not None:
        table.extend([
            ["Eval reward", fancy_float(eval_result)],
        ])
    
    table.extend([
        ["Total timesteps", total_timesteps],
        ["Sampling time (s)", fancy_float(t_sampling)],
        ["Training time (s)", fancy_float(t_training)],
        ["PMD time (s)", fancy_float(t_pmd)],])
    
    if t_eval is not None:
        assert eval_result is not None
        table.extend([
            ["Evaluation time (s)", fancy_float(t_eval)],
        ])

    table.extend([
        ["Execution time (s)", fancy_float(execution_time)],
    ])

    # Log extra metrics if provided
    if extra_metrics is not None:
        for key, value in extra_metrics.items():
            if isinstance(value, float):
                writer.add_scalar(key, value, global_step)
                # Add selected metrics to table
                if key in ['metrics/policy_entropy', 'metrics/mean_q_value', 
                           'metrics/operator_error_hs', 'metrics/reward_error_rkhs']:
                    short_key = key.replace('metrics/', '')
                    table.append([short_key, fancy_float(value)])
            elif isinstance(value, (int, bool)):
                writer.add_scalar(key, float(value), global_step)

    # Use "grid" instead of "fancy_grid" for Windows compatibility
    # fancy_grid uses Unicode characters that cause encoding errors on Windows
    import platform
    if platform.system() == "Windows":
        tablefmt = "grid"
    else:
        tablefmt = "fancy_grid"
    
    table_str = tabulate(table, headers="firstrow", tablefmt=tablefmt, numalign='right')
  
    # Log to stdout and log file
    log_file.write("\n")
    log_file.write(table_str)
    log_file.flush()
    
    # Print with error handling for Windows encoding issues
    try:
        print(table_str)
    except UnicodeEncodeError:
        # Fallback to simple format
        simple_table = tabulate(table, headers="firstrow", tablefmt="simple", numalign='right')
        print(simple_table)
