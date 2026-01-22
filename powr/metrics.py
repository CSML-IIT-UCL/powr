"""
Metrics module for POWR experiments.

This module provides functions to compute approximation errors for:
- Transition operator T
- Reward function r  
- Q-value function q

Based on Lemma 8 from the paper:
||q_hat - q||_inf <= (1/(1-gamma')) * [C_psi * ||r_n - r||_G + gamma*||r||_inf/(1-gamma) * ||T - T_n||_HS]
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ApproximationMetrics:
    """Container for approximation error metrics."""
    
    # Operator approximation error ||T - T_n||_HS
    operator_error_hs: float = 0.0
    
    # Reward approximation error ||r - r_n||_G (RKHS norm)
    reward_error_rkhs: float = 0.0
    
    # Reward approximation error ||r - r_n||_inf (uniform norm)
    reward_error_inf: float = 0.0
    
    # Q-value approximation error ||q - q_hat||_inf
    q_error_inf: float = 0.0
    
    # Q-value approximation error bound from Lemma 8
    q_error_bound: float = 0.0
    
    # Training time metrics
    kernel_eval_time: float = 0.0
    solve_time: float = 0.0
    total_train_time: float = 0.0
    
    # Dataset statistics
    n_samples: int = 0
    n_subsamples: int = 0
    
    # Policy metrics
    policy_entropy: float = 0.0
    mean_q_value: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "metrics/operator_error_hs": self.operator_error_hs,
            "metrics/reward_error_rkhs": self.reward_error_rkhs,
            "metrics/reward_error_inf": self.reward_error_inf,
            "metrics/q_error_inf": self.q_error_inf,
            "metrics/q_error_bound": self.q_error_bound,
            "metrics/kernel_eval_time": self.kernel_eval_time,
            "metrics/solve_time": self.solve_time,
            "metrics/total_train_time": self.total_train_time,
            "metrics/n_samples": float(self.n_samples),
            "metrics/n_subsamples": float(self.n_subsamples),
            "metrics/policy_entropy": self.policy_entropy,
            "metrics/mean_q_value": self.mean_q_value,
        }


def compute_operator_error_hs(
    B_exact: jnp.ndarray,
    B_approx: jnp.ndarray,
) -> float:
    """
    Compute Hilbert-Schmidt norm of operator error ||T - T_n||_HS.
    
    For finite-dimensional approximations, this is the Frobenius norm.
    
    Args:
        B_exact: Exact operator matrix (or best available approximation)
        B_approx: Approximate operator matrix
        
    Returns:
        Hilbert-Schmidt norm of the difference
    """
    diff = B_exact - B_approx
    return float(jnp.sqrt(jnp.sum(diff ** 2)))


def compute_reward_error_rkhs(
    r_exact: jnp.ndarray,
    r_approx: jnp.ndarray,
    K_sub_sub: jnp.ndarray,
) -> float:
    """
    Compute RKHS norm of reward error ||r - r_n||_G.
    
    ||f||_G^2 = f^T K^{-1} f for f = sum_i alpha_i k(., x_i)
    
    For computational stability, we use ||r - r_n||_2 as a proxy
    when exact computation is not feasible.
    
    Args:
        r_exact: Exact reward weights
        r_approx: Approximate reward weights
        K_sub_sub: Kernel matrix on subsample points
        
    Returns:
        RKHS norm of reward error (or L2 proxy)
    """
    diff = r_exact - r_approx
    
    try:
        # Try to compute exact RKHS norm
        # ||f||_G^2 = alpha^T K alpha where f = K @ alpha
        # So we need to solve K @ alpha = diff, then compute alpha^T @ diff
        alpha = jnp.linalg.solve(K_sub_sub + 1e-6 * jnp.eye(K_sub_sub.shape[0]), diff)
        rkhs_norm_sq = float(jnp.dot(alpha.flatten(), diff.flatten()))
        if rkhs_norm_sq >= 0:
            return float(jnp.sqrt(rkhs_norm_sq))
    except:
        pass
    
    # Fallback to L2 norm
    return float(jnp.linalg.norm(diff))


def compute_reward_error_inf(
    r_exact: jnp.ndarray,
    r_approx: jnp.ndarray,
) -> float:
    """
    Compute uniform norm of reward error ||r - r_n||_inf.
    
    Args:
        r_exact: Exact reward values
        r_approx: Approximate reward values
        
    Returns:
        Maximum absolute difference
    """
    return float(jnp.max(jnp.abs(r_exact - r_approx)))


def compute_q_error_inf(
    q_exact: jnp.ndarray,
    q_approx: jnp.ndarray,
) -> float:
    """
    Compute uniform norm of Q-value error ||q - q_hat||_inf.
    
    Args:
        q_exact: Exact Q-values
        q_approx: Approximate Q-values
        
    Returns:
        Maximum absolute difference
    """
    return float(jnp.max(jnp.abs(q_exact - q_approx)))


def compute_q_error_bound(
    reward_error_rkhs: float,
    operator_error_hs: float,
    gamma: float,
    gamma_prime: float,
    r_inf_norm: float,
    C_psi: float = 1.0,
) -> float:
    """
    Compute Q-value error bound from Lemma 8.
    
    ||q_hat - q||_inf <= (1/(1-gamma')) * [C_psi * ||r_n - r||_G + gamma*||r||_inf/(1-gamma) * ||T - T_n||_HS]
    
    Args:
        reward_error_rkhs: ||r - r_n||_G
        operator_error_hs: ||T - T_n||_HS
        gamma: Discount factor
        gamma_prime: gamma' such that gamma * ||T_n|| < gamma' < 1
        r_inf_norm: ||r||_inf
        C_psi: Kernel bound constant
        
    Returns:
        Upper bound on Q-value error
    """
    if gamma_prime >= 1.0:
        gamma_prime = 0.999
        
    term1 = C_psi * reward_error_rkhs
    term2 = (gamma * r_inf_norm / (1 - gamma)) * operator_error_hs
    
    return (1 / (1 - gamma_prime)) * (term1 + term2)


def compute_policy_entropy(
    policy_probs: jnp.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute entropy of policy distribution.
    
    H(pi) = -sum_a pi(a|s) log pi(a|s)
    
    Args:
        policy_probs: Policy probabilities, shape (n_states, n_actions)
        eps: Small constant for numerical stability
        
    Returns:
        Mean entropy across states
    """
    # Clip probabilities for numerical stability
    probs = jnp.clip(policy_probs, eps, 1.0 - eps)
    
    # Compute entropy per state
    entropy_per_state = -jnp.sum(probs * jnp.log(probs), axis=-1)
    
    return float(jnp.mean(entropy_per_state))


def compute_kernel_approximation_error(
    kernel_exact: callable,
    kernel_approx: callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute kernel approximation error statistics.
    
    Args:
        kernel_exact: Exact kernel function
        kernel_approx: Approximate kernel function
        X: First set of points
        Y: Second set of points
        
    Returns:
        Tuple of (mean_error, max_error, relative_error)
    """
    K_exact = kernel_exact(X, Y)
    K_approx = kernel_approx(X, Y)
    
    diff = jnp.abs(K_exact - K_approx)
    
    mean_error = float(jnp.mean(diff))
    max_error = float(jnp.max(diff))
    
    # Relative error
    denom = jnp.abs(K_exact) + 1e-10
    relative_error = float(jnp.mean(diff / denom))
    
    return mean_error, max_error, relative_error


class MetricsTracker:
    """
    Tracks metrics across training epochs.
    """
    
    def __init__(self, log_to_wandb: bool = True, log_to_tensorboard: bool = True):
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        self.history: Dict[str, list] = {}
        self.current_epoch: int = 0
        
    def log(self, metrics: ApproximationMetrics, epoch: int, writer=None):
        """Log metrics for current epoch."""
        self.current_epoch = epoch
        metrics_dict = metrics.to_dict()
        
        # Store in history
        for key, value in metrics_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((epoch, value))
        
        # Log to tensorboard
        if self.log_to_tensorboard and writer is not None:
            for key, value in metrics_dict.items():
                writer.add_scalar(key, value, epoch)
        
        # Log to wandb
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics_dict, step=epoch)
            except ImportError:
                pass
                
    def get_history(self, metric_name: str) -> list:
        """Get history for a specific metric."""
        return self.history.get(metric_name, [])
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values for all metrics."""
        return {
            key: values[-1][1] if values else 0.0
            for key, values in self.history.items()
        }


def compute_monte_carlo_q_estimate(
    env,
    policy_fn: callable,
    state: jnp.ndarray,
    action: int,
    gamma: float,
    n_rollouts: int = 100,
    max_steps: int = 200,
) -> float:
    """
    Estimate Q(s, a) using Monte Carlo rollouts.
    
    This provides a "ground truth" estimate for comparison.
    
    Args:
        env: Gym environment
        policy_fn: Function that returns action probabilities given state
        state: Starting state
        action: Starting action
        gamma: Discount factor
        n_rollouts: Number of rollouts to average
        max_steps: Maximum steps per rollout
        
    Returns:
        Monte Carlo estimate of Q(s, a)
    """
    import numpy as np
    
    q_estimates = []
    
    for _ in range(n_rollouts):
        # Reset to initial state (if possible)
        try:
            env.reset()
            env.unwrapped.state = np.array(state)
        except:
            env.reset()
            
        total_return = 0.0
        discount = 1.0
        
        # Take initial action
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        discount *= gamma
        
        # Continue with policy
        current_state = next_state
        for step in range(max_steps - 1):
            if terminated or truncated:
                break
                
            # Sample action from policy
            probs = policy_fn(current_state)
            action = np.random.choice(len(probs), p=np.array(probs).flatten())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_return += discount * reward
            discount *= gamma
            current_state = next_state
            
        q_estimates.append(total_return)
        
    return float(np.mean(q_estimates))
