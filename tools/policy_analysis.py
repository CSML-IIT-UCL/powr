#!/usr/bin/env python3
"""
Policy Analysis Tool for POWR.

This script provides tools for analyzing:
1. PMD convergence tracking per Theorem 7
2. Policy entropy evolution
3. Q-value distribution analysis
4. Action distribution at key states
5. Monte Carlo Q-value validation

Usage:
    python tools/policy_analysis.py --checkpoint runs/.../checkpoint.pkl
    python tools/policy_analysis.py --run-and-analyze --env MountainCar-v0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax.numpy as jnp
import gymnasium as gym

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# PMD Convergence Tracking
# ============================================================================

@dataclass
class PMDConvergenceMetrics:
    """Metrics for tracking PMD convergence."""
    epoch: int
    policy_entropy: float
    mean_q_value: float
    max_q_value: float
    min_q_value: float
    q_value_std: float
    eval_reward: float
    train_reward: float
    estimated_optimality_gap: float = 0.0


class PMDConvergenceTracker:
    """
    Track PMD convergence based on Theorem 7.
    
    Theorem 7 states:
    J(π*) - J(πT) ≤ εT + O(1/T + 1/T * Σ εt)
    
    We track proxies for these quantities since we don't have access to π*.
    """
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.metrics_history: List[PMDConvergenceMetrics] = []
        self.best_reward = float('-inf')
        
    def add_metrics(self, metrics: PMDConvergenceMetrics):
        """Add metrics for current epoch."""
        self.metrics_history.append(metrics)
        
        if metrics.eval_reward > self.best_reward:
            self.best_reward = metrics.eval_reward
            
    def compute_estimated_optimality_gap(self, current_reward: float) -> float:
        """
        Estimate optimality gap using best observed reward as proxy for J(π*).
        
        This is a lower bound on the true gap.
        """
        return max(0, self.best_reward - current_reward)
    
    def compute_convergence_rate(self) -> Optional[float]:
        """
        Estimate convergence rate from reward history.
        
        Returns estimated rate or None if insufficient data.
        """
        if len(self.metrics_history) < 10:
            return None
            
        rewards = [m.eval_reward for m in self.metrics_history]
        epochs = list(range(len(rewards)))
        
        # Fit 1/T convergence model: gap ≈ c/T
        # Log transform: log(gap) ≈ log(c) - log(T)
        gaps = [max(0.01, self.best_reward - r) for r in rewards]
        
        # Use linear regression on log-log scale
        log_t = np.log(np.array(epochs) + 1)
        log_gap = np.log(gaps)
        
        # Fit slope
        slope = np.polyfit(log_t, log_gap, 1)[0]
        
        return -slope  # Negative slope indicates convergence
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}
            
        rewards = [m.eval_reward for m in self.metrics_history]
        entropies = [m.policy_entropy for m in self.metrics_history]
        
        return {
            'n_epochs': len(self.metrics_history),
            'final_reward': rewards[-1],
            'best_reward': self.best_reward,
            'mean_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'final_entropy': entropies[-1],
            'initial_entropy': entropies[0] if entropies else 0,
            'entropy_decay': (entropies[0] - entropies[-1]) if entropies else 0,
            'convergence_rate': self.compute_convergence_rate(),
        }
    
    def save(self, filepath: str):
        """Save tracking history."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'metrics_history': self.metrics_history,
                'best_reward': self.best_reward,
                'gamma': self.gamma,
            }, f)
            
    def load(self, filepath: str):
        """Load tracking history."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.metrics_history = data['metrics_history']
            self.best_reward = data['best_reward']
            self.gamma = data.get('gamma', 0.99)


# ============================================================================
# Policy Quality Analysis
# ============================================================================

def compute_policy_entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute entropy of policy distribution.
    
    H(π) = -Σ π(a|s) log π(a|s)
    """
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(probs * np.log(probs))


def analyze_action_distribution(
    mdp_manager,
    states: np.ndarray,
    state_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze action distribution at given states.
    
    Args:
        mdp_manager: MDPManager instance
        states: States to analyze
        state_labels: Optional labels for states
        
    Returns:
        Dictionary with action distribution analysis
    """
    analysis = {
        'states': states.tolist() if hasattr(states, 'tolist') else states,
        'action_probs': [],
        'entropies': [],
        'max_actions': [],
    }
    
    for i, state in enumerate(states):
        state_arr = np.array(state).reshape(1, -1)
        probs = mdp_manager.evaluate_pi(state_arr)
        probs = np.array(probs).flatten()
        
        analysis['action_probs'].append(probs.tolist())
        analysis['entropies'].append(compute_policy_entropy(probs))
        analysis['max_actions'].append(int(np.argmax(probs)))
        
    analysis['mean_entropy'] = np.mean(analysis['entropies'])
    analysis['min_entropy'] = np.min(analysis['entropies'])
    analysis['max_entropy'] = np.max(analysis['entropies'])
    
    return analysis


def monte_carlo_q_estimate(
    env,
    policy_fn,
    state,
    action: int,
    gamma: float = 0.99,
    n_rollouts: int = 100,
    max_steps: int = 200,
) -> Tuple[float, float]:
    """
    Estimate Q(s, a) using Monte Carlo rollouts.
    
    Returns:
        (mean_return, std_return)
    """
    returns = []
    
    for _ in range(n_rollouts):
        total_return = 0.0
        discount = 1.0
        
        # Reset and set initial state if possible
        obs, _ = env.reset()
        try:
            env.unwrapped.state = np.array(state)
            obs = state
        except:
            pass
        
        # Take initial action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        discount *= gamma
        
        # Follow policy
        current_obs = next_obs
        for step in range(max_steps - 1):
            if terminated or truncated:
                break
                
            probs = policy_fn(current_obs)
            probs = np.array(probs).flatten()
            probs = probs / probs.sum()  # Normalize
            
            try:
                action = np.random.choice(len(probs), p=probs)
            except:
                action = np.argmax(probs)
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_return += discount * reward
            discount *= gamma
            current_obs = next_obs
            
        returns.append(total_return)
        
    return np.mean(returns), np.std(returns)


def validate_q_values(
    mdp_manager,
    env,
    n_states: int = 20,
    n_rollouts: int = 50,
    gamma: float = 0.99,
) -> Dict[str, Any]:
    """
    Validate learned Q-values against Monte Carlo estimates.
    
    Args:
        mdp_manager: MDPManager instance
        env: Gym environment
        n_states: Number of random states to sample
        n_rollouts: Rollouts per state-action pair for MC estimate
        gamma: Discount factor
        
    Returns:
        Validation results
    """
    results = {
        'states': [],
        'actions': [],
        'learned_q': [],
        'mc_q_mean': [],
        'mc_q_std': [],
        'errors': [],
    }
    
    # Sample random states by running environment
    obs, _ = env.reset()
    states_collected = []
    
    for _ in range(n_states * 10):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        if len(states_collected) < n_states:
            states_collected.append(obs.copy() if hasattr(obs, 'copy') else obs)
            
        if terminated or truncated:
            obs, _ = env.reset()
            
    # For each state, validate Q-values
    n_actions = env.action_space.n
    
    for state in states_collected[:n_states]:
        state_arr = np.array(state).reshape(1, -1)
        
        # Get learned Q-values
        learned_q = mdp_manager.compute_q_values_at_states(jnp.array(state_arr))
        learned_q = np.array(learned_q).flatten()
        
        # Get MC estimates for each action
        for action in range(min(n_actions, 3)):  # Limit actions for speed
            mc_mean, mc_std = monte_carlo_q_estimate(
                env,
                lambda s: mdp_manager.evaluate_pi(np.array(s).reshape(1, -1)),
                state,
                action,
                gamma=gamma,
                n_rollouts=n_rollouts,
            )
            
            results['states'].append(state.tolist() if hasattr(state, 'tolist') else state)
            results['actions'].append(action)
            results['learned_q'].append(float(learned_q[action]))
            results['mc_q_mean'].append(mc_mean)
            results['mc_q_std'].append(mc_std)
            results['errors'].append(abs(float(learned_q[action]) - mc_mean))
            
    # Summary statistics
    results['mean_error'] = np.mean(results['errors'])
    results['max_error'] = np.max(results['errors'])
    results['correlation'] = np.corrcoef(results['learned_q'], results['mc_q_mean'])[0, 1]
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_convergence(tracker: PMDConvergenceTracker, output_path: str):
    """Plot PMD convergence metrics."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
        
    if not tracker.metrics_history:
        print("No metrics to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = [m.epoch for m in tracker.metrics_history]
    rewards = [m.eval_reward for m in tracker.metrics_history]
    entropies = [m.policy_entropy for m in tracker.metrics_history]
    mean_q = [m.mean_q_value for m in tracker.metrics_history]
    gaps = [tracker.compute_estimated_optimality_gap(r) for r in rewards]
    
    # Reward over time
    ax = axes[0, 0]
    ax.plot(epochs, rewards, linewidth=2)
    ax.axhline(y=tracker.best_reward, color='r', linestyle='--', label='Best')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Optimality gap
    ax = axes[0, 1]
    ax.plot(epochs, gaps, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Estimated Optimality Gap')
    ax.set_title('Convergence to Optimum')
    ax.grid(True, alpha=0.3)
    
    # Policy entropy
    ax = axes[1, 0]
    ax.plot(epochs, entropies, linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Policy Entropy')
    ax.set_title('Policy Entropy Evolution')
    ax.grid(True, alpha=0.3)
    
    # Mean Q-value
    ax = axes[1, 1]
    ax.plot(epochs, mean_q, linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Q-value')
    ax.set_title('Q-value Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_q_validation(validation_results: Dict, output_path: str):
    """Plot Q-value validation results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Learned vs MC Q-values
    ax = axes[0]
    learned = validation_results['learned_q']
    mc_mean = validation_results['mc_q_mean']
    mc_std = validation_results['mc_q_std']
    
    ax.errorbar(learned, mc_mean, yerr=mc_std, fmt='o', capsize=3, alpha=0.7)
    
    # Add diagonal line
    min_val = min(min(learned), min(mc_mean))
    max_val = max(max(learned), max(mc_mean))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax.set_xlabel('Learned Q-value')
    ax.set_ylabel('Monte Carlo Q-value')
    ax.set_title(f'Q-value Validation (corr={validation_results["correlation"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1]
    errors = validation_results['errors']
    ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(errors), color='r', linestyle='--', 
               label=f'Mean={np.mean(errors):.3f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Q-value Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Policy Analysis Tool")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to analyze",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCar-v0",
        help="Environment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/policy_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--validate-q",
        action="store_true",
        help="Run Monte Carlo Q-value validation",
    )
    parser.add_argument(
        "--n-mc-rollouts",
        type=int,
        default=50,
        help="Number of MC rollouts for Q validation",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPolicy Analysis Tool")
    print(f"{'='*50}")
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        
        # Load checkpoint and MDPManager
        from powr.utils import load_checkpoint
        from powr.MDPManager import MDPManager
        from powr.kernels import gaussian_kernel_diag
        
        checkpoint_data = load_checkpoint(args.checkpoint)
        
        # Create environment
        env = gym.make(args.env)
        
        # Create MDPManager (this is a simplified version)
        # In practice, you would load the full state
        print("Checkpoint analysis requires full MDPManager state...")
        print("Use --validate-q with a live run for Q-value validation")
        
    else:
        print("No checkpoint provided. Running demonstration analysis...")
        
        # Create tracker with synthetic data for demonstration
        tracker = PMDConvergenceTracker(gamma=0.99)
        
        # Generate synthetic convergence data
        np.random.seed(42)
        for epoch in range(100):
            # Simulate improving rewards with noise
            base_reward = -200 + 100 * (1 - np.exp(-epoch / 30))
            noise = np.random.normal(0, 10)
            
            # Simulate decreasing entropy
            entropy = 1.5 * np.exp(-epoch / 50) + 0.1
            
            metrics = PMDConvergenceMetrics(
                epoch=epoch,
                policy_entropy=entropy,
                mean_q_value=-50 + 30 * (1 - np.exp(-epoch / 20)),
                max_q_value=-30 + 20 * (1 - np.exp(-epoch / 20)),
                min_q_value=-100 + 50 * (1 - np.exp(-epoch / 30)),
                q_value_std=20 * np.exp(-epoch / 40) + 5,
                eval_reward=base_reward + noise,
                train_reward=base_reward + np.random.normal(0, 15),
            )
            tracker.add_metrics(metrics)
        
        # Print summary
        summary = tracker.get_summary()
        print("\nConvergence Summary:")
        print(f"  Epochs: {summary['n_epochs']}")
        print(f"  Final reward: {summary['final_reward']:.2f}")
        print(f"  Best reward: {summary['best_reward']:.2f}")
        print(f"  Mean reward: {summary['mean_reward']:.2f}")
        print(f"  Initial entropy: {summary['initial_entropy']:.3f}")
        print(f"  Final entropy: {summary['final_entropy']:.3f}")
        if summary['convergence_rate']:
            print(f"  Estimated convergence rate: {summary['convergence_rate']:.3f}")
        
        # Generate plots
        plot_convergence(tracker, str(output_dir / 'convergence_plot.png'))
        
        # Save tracker
        tracker.save(str(output_dir / 'convergence_tracker.pkl'))
        
        # Generate synthetic Q-validation results
        print("\nGenerating synthetic Q-value validation...")
        n_samples = 30
        validation_results = {
            'states': [list(np.random.randn(2)) for _ in range(n_samples)],
            'actions': [i % 3 for i in range(n_samples)],
            'learned_q': list(np.random.randn(n_samples) * 10 - 50),
            'mc_q_mean': [],
            'mc_q_std': list(np.abs(np.random.randn(n_samples) * 5)),
            'errors': [],
        }
        
        # Add correlated MC values with noise
        for q in validation_results['learned_q']:
            mc = q + np.random.randn() * 8
            validation_results['mc_q_mean'].append(mc)
            validation_results['errors'].append(abs(q - mc))
            
        validation_results['mean_error'] = np.mean(validation_results['errors'])
        validation_results['max_error'] = np.max(validation_results['errors'])
        validation_results['correlation'] = np.corrcoef(
            validation_results['learned_q'], 
            validation_results['mc_q_mean']
        )[0, 1]
        
        plot_q_validation(validation_results, str(output_dir / 'q_validation_plot.png'))
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
