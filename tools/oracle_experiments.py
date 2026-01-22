#!/usr/bin/env python3
"""
Oracle experiments for error decomposition in POWR.

This script implements experiments to identify whether the transition operator T
or the reward function r is the bottleneck in learning.

Based on Lemma 8 from the paper:
||q_hat - q||_inf <= (1/(1-gamma')) * [C_psi * ||r_n - r||_G + gamma*||r||_inf/(1-gamma) * ||T - T_n||_HS]

Experiments:
1. Oracle T + learned r: Use ground-truth transition operator, learn reward
2. Oracle r + learned T: Use ground-truth reward, learn transition operator
3. Compare errors to identify which component dominates

This works best on discrete/tabular environments where we can compute exact operators.
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
# Ground Truth Computation for Tabular MDPs
# ============================================================================

def compute_ground_truth_tabular(env_name: str) -> Dict[str, Any]:
    """
    Compute ground truth transition operator T and reward function r
    for tabular environments.
    
    Args:
        env_name: Name of the environment (FrozenLake-v1, Taxi-v3)
        
    Returns:
        Dictionary containing:
        - T: Transition matrix (n_states * n_actions, n_states)
        - r: Reward vector (n_states * n_actions,)
        - n_states: Number of states
        - n_actions: Number of actions
    """
    if env_name == "FrozenLake-v1":
        env = gym.make("FrozenLake-v1", is_slippery=False)
    elif env_name == "Taxi-v3":
        env = gym.make("Taxi-v3")
    else:
        raise ValueError(f"Unsupported environment for oracle: {env_name}")
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize transition matrix and reward vector
    T = np.zeros((n_states * n_actions, n_states))
    r = np.zeros(n_states * n_actions)
    
    # Build transition matrix from environment dynamics
    for s in range(n_states):
        for a in range(n_actions):
            idx = s * n_actions + a
            
            # Get transitions from P
            if hasattr(env.unwrapped, 'P'):
                transitions = env.unwrapped.P[s][a]
                for prob, next_state, reward, done in transitions:
                    T[idx, next_state] += prob
                    r[idx] += prob * reward
            else:
                # Sample-based approximation
                env.reset()
                env.unwrapped.s = s
                next_state, reward, _, _, _ = env.step(a)
                T[idx, next_state] = 1.0
                r[idx] = reward
    
    env.close()
    
    return {
        'T': T,
        'r': r,
        'n_states': n_states,
        'n_actions': n_actions,
        'env_name': env_name,
    }


def compute_exact_q_values(
    T: np.ndarray,
    r: np.ndarray,
    n_states: int,
    n_actions: int,
    gamma: float = 0.99,
    policy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute exact Q-values for a policy using the Bellman equation.
    
    Q(s,a) = r(s,a) + gamma * sum_s' T(s'|s,a) * V(s')
    V(s) = sum_a pi(a|s) * Q(s,a)
    
    Args:
        T: Transition matrix (n_states * n_actions, n_states)
        r: Reward vector (n_states * n_actions,)
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor
        policy: Policy matrix (n_states, n_actions), uniform if None
        
    Returns:
        Q-values (n_states, n_actions)
    """
    if policy is None:
        policy = np.ones((n_states, n_actions)) / n_actions
    
    # Reshape r to (n_states, n_actions)
    r_sa = r.reshape(n_states, n_actions)
    
    # Reshape T to (n_states, n_actions, n_states)
    T_sa = T.reshape(n_states, n_actions, n_states)
    
    # Build policy-weighted transition matrix P_pi: (n_states, n_states)
    # P_pi[s, s'] = sum_a pi(a|s) * T(s'|s,a)
    P_pi = np.einsum('sa,sab->sb', policy, T_sa)
    
    # Build reward under policy r_pi: (n_states,)
    r_pi = np.einsum('sa,sa->s', policy, r_sa)
    
    # Solve for V: V = (I - gamma * P_pi)^{-1} * r_pi
    I = np.eye(n_states)
    V = np.linalg.solve(I - gamma * P_pi, r_pi)
    
    # Compute Q: Q(s,a) = r(s,a) + gamma * sum_s' T(s'|s,a) * V(s')
    Q = r_sa + gamma * np.einsum('sab,b->sa', T_sa, V)
    
    return Q


# ============================================================================
# Oracle Experiment Classes
# ============================================================================

@dataclass
class OracleExperimentResult:
    """Results from an oracle experiment."""
    experiment_type: str  # "oracle_T", "oracle_r", "baseline"
    q_error_inf: float = 0.0
    q_error_mse: float = 0.0
    operator_error_hs: float = 0.0
    reward_error_rkhs: float = 0.0
    reward_error_inf: float = 0.0
    training_epochs: int = 0
    final_reward: float = 0.0
    error_history: List[Dict[str, float]] = field(default_factory=list)


class OracleExperiment:
    """
    Run oracle experiments to decompose approximation errors.
    """
    
    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        gamma: float = 0.99,
        n_samples: int = 1000,
        seed: int = 0,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.n_samples = n_samples
        self.seed = seed
        
        # Compute ground truth
        print(f"Computing ground truth for {env_name}...")
        self.ground_truth = compute_ground_truth_tabular(env_name)
        
        # Compute exact Q-values for uniform policy
        self.exact_Q = compute_exact_q_values(
            self.ground_truth['T'],
            self.ground_truth['r'],
            self.ground_truth['n_states'],
            self.ground_truth['n_actions'],
            gamma=gamma,
        )
        
        print(f"Ground truth computed: {self.ground_truth['n_states']} states, "
              f"{self.ground_truth['n_actions']} actions")
        
    def collect_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect samples from the environment.
        
        Returns:
            (states, actions, next_states, rewards)
        """
        np.random.seed(self.seed)
        
        if self.env_name == "FrozenLake-v1":
            env = gym.make("FrozenLake-v1", is_slippery=False)
        else:
            env = gym.make(self.env_name)
            
        states = []
        actions = []
        next_states = []
        rewards = []
        
        state, _ = env.reset(seed=self.seed)
        
        for _ in range(self.n_samples):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
                
        env.close()
        
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards),
        )
    
    def learn_transition_operator(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        la: float = 1e-6,
    ) -> np.ndarray:
        """
        Learn transition operator from samples using least squares.
        
        Returns:
            Learned transition matrix T_hat
        """
        n_states = self.ground_truth['n_states']
        n_actions = self.ground_truth['n_actions']
        
        # Build empirical transition counts
        T_hat = np.zeros((n_states * n_actions, n_states))
        counts = np.zeros(n_states * n_actions)
        
        for s, a, s_next in zip(states, actions, next_states):
            idx = s * n_actions + a
            T_hat[idx, s_next] += 1
            counts[idx] += 1
            
        # Normalize to get probabilities (with regularization)
        for idx in range(n_states * n_actions):
            if counts[idx] > 0:
                T_hat[idx] /= counts[idx]
            else:
                # Uniform prior for unvisited state-actions
                T_hat[idx] = 1.0 / n_states
                
        return T_hat
    
    def learn_reward_function(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        la: float = 1e-6,
    ) -> np.ndarray:
        """
        Learn reward function from samples.
        
        Returns:
            Learned reward vector r_hat
        """
        n_states = self.ground_truth['n_states']
        n_actions = self.ground_truth['n_actions']
        
        # Build empirical reward averages
        r_hat = np.zeros(n_states * n_actions)
        counts = np.zeros(n_states * n_actions)
        
        for s, a, r in zip(states, actions, rewards):
            idx = s * n_actions + a
            r_hat[idx] += r
            counts[idx] += 1
            
        # Average rewards (with regularization for unvisited)
        for idx in range(n_states * n_actions):
            if counts[idx] > 0:
                r_hat[idx] /= counts[idx]
            # else: leave as 0 (unknown reward)
                
        return r_hat
    
    def run_baseline_experiment(self) -> OracleExperimentResult:
        """
        Run baseline experiment: learn both T and r from data.
        """
        print("\n--- Baseline Experiment: Learn both T and r ---")
        
        # Collect samples
        states, actions, next_states, rewards = self.collect_samples()
        
        # Learn both components
        T_hat = self.learn_transition_operator(states, actions, next_states)
        r_hat = self.learn_reward_function(states, actions, rewards)
        
        # Compute errors
        T_true = self.ground_truth['T']
        r_true = self.ground_truth['r']
        
        operator_error = np.sqrt(np.sum((T_true - T_hat) ** 2))
        reward_error_inf = np.max(np.abs(r_true - r_hat))
        reward_error_mse = np.sqrt(np.mean((r_true - r_hat) ** 2))
        
        # Compute Q-values with learned components
        Q_hat = compute_exact_q_values(
            T_hat, r_hat,
            self.ground_truth['n_states'],
            self.ground_truth['n_actions'],
            gamma=self.gamma,
        )
        
        q_error_inf = np.max(np.abs(self.exact_Q - Q_hat))
        q_error_mse = np.sqrt(np.mean((self.exact_Q - Q_hat) ** 2))
        
        result = OracleExperimentResult(
            experiment_type="baseline",
            q_error_inf=q_error_inf,
            q_error_mse=q_error_mse,
            operator_error_hs=operator_error,
            reward_error_rkhs=reward_error_mse,
            reward_error_inf=reward_error_inf,
        )
        
        print(f"  Operator error (HS): {operator_error:.6f}")
        print(f"  Reward error (inf): {reward_error_inf:.6f}")
        print(f"  Q-value error (inf): {q_error_inf:.6f}")
        
        return result
    
    def run_oracle_T_experiment(self) -> OracleExperimentResult:
        """
        Run oracle T experiment: use true T, learn r from data.
        """
        print("\n--- Oracle T Experiment: True T + Learned r ---")
        
        # Collect samples
        states, actions, next_states, rewards = self.collect_samples()
        
        # Use ground truth T, learn r
        T_true = self.ground_truth['T']
        r_hat = self.learn_reward_function(states, actions, rewards)
        
        # Compute errors
        r_true = self.ground_truth['r']
        reward_error_inf = np.max(np.abs(r_true - r_hat))
        reward_error_mse = np.sqrt(np.mean((r_true - r_hat) ** 2))
        
        # Compute Q-values with true T and learned r
        Q_hat = compute_exact_q_values(
            T_true, r_hat,
            self.ground_truth['n_states'],
            self.ground_truth['n_actions'],
            gamma=self.gamma,
        )
        
        q_error_inf = np.max(np.abs(self.exact_Q - Q_hat))
        q_error_mse = np.sqrt(np.mean((self.exact_Q - Q_hat) ** 2))
        
        result = OracleExperimentResult(
            experiment_type="oracle_T",
            q_error_inf=q_error_inf,
            q_error_mse=q_error_mse,
            operator_error_hs=0.0,  # Using true T
            reward_error_rkhs=reward_error_mse,
            reward_error_inf=reward_error_inf,
        )
        
        print(f"  Reward error (inf): {reward_error_inf:.6f}")
        print(f"  Q-value error (inf): {q_error_inf:.6f}")
        print(f"  (Operator error is 0 - using true T)")
        
        return result
    
    def run_oracle_r_experiment(self) -> OracleExperimentResult:
        """
        Run oracle r experiment: use true r, learn T from data.
        """
        print("\n--- Oracle r Experiment: Learned T + True r ---")
        
        # Collect samples
        states, actions, next_states, rewards = self.collect_samples()
        
        # Learn T, use ground truth r
        T_hat = self.learn_transition_operator(states, actions, next_states)
        r_true = self.ground_truth['r']
        
        # Compute errors
        T_true = self.ground_truth['T']
        operator_error = np.sqrt(np.sum((T_true - T_hat) ** 2))
        
        # Compute Q-values with learned T and true r
        Q_hat = compute_exact_q_values(
            T_hat, r_true,
            self.ground_truth['n_states'],
            self.ground_truth['n_actions'],
            gamma=self.gamma,
        )
        
        q_error_inf = np.max(np.abs(self.exact_Q - Q_hat))
        q_error_mse = np.sqrt(np.mean((self.exact_Q - Q_hat) ** 2))
        
        result = OracleExperimentResult(
            experiment_type="oracle_r",
            q_error_inf=q_error_inf,
            q_error_mse=q_error_mse,
            operator_error_hs=operator_error,
            reward_error_rkhs=0.0,  # Using true r
            reward_error_inf=0.0,
        )
        
        print(f"  Operator error (HS): {operator_error:.6f}")
        print(f"  Q-value error (inf): {q_error_inf:.6f}")
        print(f"  (Reward error is 0 - using true r)")
        
        return result
    
    def run_all_experiments(self) -> Dict[str, OracleExperimentResult]:
        """Run all oracle experiments and compare."""
        results = {
            'baseline': self.run_baseline_experiment(),
            'oracle_T': self.run_oracle_T_experiment(),
            'oracle_r': self.run_oracle_r_experiment(),
        }
        
        # Analysis
        print("\n" + "=" * 60)
        print("Error Decomposition Analysis")
        print("=" * 60)
        
        baseline_q = results['baseline'].q_error_inf
        oracle_T_q = results['oracle_T'].q_error_inf
        oracle_r_q = results['oracle_r'].q_error_inf
        
        print(f"\nQ-value errors:")
        print(f"  Baseline (learn both): {baseline_q:.6f}")
        print(f"  Oracle T (learn r):    {oracle_T_q:.6f}")
        print(f"  Oracle r (learn T):    {oracle_r_q:.6f}")
        
        # Determine bottleneck
        if oracle_T_q < oracle_r_q:
            print(f"\n=> Bottleneck: TRANSITION OPERATOR (T)")
            print(f"   Error reduced by {(1 - oracle_r_q/baseline_q)*100:.1f}% when using true T")
        else:
            print(f"\n=> Bottleneck: REWARD FUNCTION (r)")
            print(f"   Error reduced by {(1 - oracle_T_q/baseline_q)*100:.1f}% when using true r")
        
        # Contribution analysis based on Lemma 8
        print(f"\nComponent contributions (based on Lemma 8):")
        print(f"  Operator error contribution: {results['baseline'].operator_error_hs:.6f}")
        print(f"  Reward error contribution:   {results['baseline'].reward_error_inf:.6f}")
        
        return results
    
    def run_sample_size_sweep(
        self,
        sample_sizes: List[int] = [100, 500, 1000, 2000, 5000],
    ) -> Dict[int, Dict[str, OracleExperimentResult]]:
        """
        Run experiments across different sample sizes to see error scaling.
        """
        all_results = {}
        
        original_n_samples = self.n_samples
        
        for n in sample_sizes:
            print(f"\n{'='*60}")
            print(f"Sample size: {n}")
            print(f"{'='*60}")
            
            self.n_samples = n
            all_results[n] = self.run_all_experiments()
            
        self.n_samples = original_n_samples
        
        return all_results


def plot_error_decomposition(results: Dict[str, OracleExperimentResult], output_path: str):
    """Plot error decomposition results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-value errors
    ax1 = axes[0]
    experiments = ['baseline', 'oracle_T', 'oracle_r']
    q_errors = [results[exp].q_error_inf for exp in experiments]
    labels = ['Learn both', 'True T + Learn r', 'Learn T + True r']
    
    bars = ax1.bar(labels, q_errors, color=['steelblue', 'coral', 'seagreen'])
    ax1.set_ylabel('Q-value Error (inf norm)')
    ax1.set_title('Q-value Approximation Error')
    ax1.tick_params(axis='x', rotation=15)
    
    # Component errors
    ax2 = axes[1]
    component_labels = ['Operator Error', 'Reward Error']
    component_values = [
        results['baseline'].operator_error_hs,
        results['baseline'].reward_error_inf,
    ]
    
    bars = ax2.bar(component_labels, component_values, color=['coral', 'seagreen'])
    ax2.set_ylabel('Error')
    ax2.set_title('Component Errors (Baseline)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_sample_size_sweep(
    results: Dict[int, Dict[str, OracleExperimentResult]], 
    output_path: str
):
    """Plot error vs sample size."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sample_sizes = sorted(results.keys())
    
    for exp_type, color, marker in [
        ('baseline', 'steelblue', 'o'),
        ('oracle_T', 'coral', 's'),
        ('oracle_r', 'seagreen', '^'),
    ]:
        errors = [results[n][exp_type].q_error_inf for n in sample_sizes]
        label = {
            'baseline': 'Learn both',
            'oracle_T': 'True T + Learn r',
            'oracle_r': 'Learn T + True r',
        }[exp_type]
        ax.plot(sample_sizes, errors, marker=marker, label=label, 
                linewidth=2, markersize=8, color=color)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Q-value Error (inf norm)')
    ax.set_title('Q-value Error vs Sample Size')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Oracle Experiments for Error Decomposition")
    
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        choices=["FrozenLake-v1", "Taxi-v3"],
        help="Environment to use (must be tabular)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--sample-sweep",
        action="store_true",
        help="Run sample size sweep experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/oracle",
        help="Output directory for results",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment
    experiment = OracleExperiment(
        env_name=args.env,
        gamma=args.gamma,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    
    if args.sample_sweep:
        # Run sample size sweep
        results = experiment.run_sample_size_sweep()
        
        # Save results
        with open(output_dir / 'sample_sweep_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Plot
        plot_sample_size_sweep(
            results, 
            str(output_dir / 'sample_sweep_plot.png')
        )
    else:
        # Run single experiment
        results = experiment.run_all_experiments()
        
        # Save results
        with open(output_dir / 'oracle_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Plot
        plot_error_decomposition(
            results,
            str(output_dir / 'error_decomposition_plot.png')
        )
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
