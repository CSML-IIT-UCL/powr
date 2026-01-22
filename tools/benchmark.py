#!/usr/bin/env python3
"""
Benchmark script for POWR experiments.

This script provides infrastructure for:
1. Running parameter sweeps (RFF features, Nystrom subsamples, kernels, etc.)
2. Collecting and aggregating metrics
3. Generating plots and reports

Usage:
    python tools/benchmark.py --experiment rff_sweep --env MountainCar-v0
    python tools/benchmark.py --experiment kernel_comparison --env CartPole-v1
    python tools/benchmark.py --experiment nystrom_sweep --env FrozenLake-v1
"""

import os
import sys
import json
import time
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import subprocess
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, some analysis features disabled")


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    env: str = "MountainCar-v0"
    kernel_method: str = "exact"  # "exact" or "rff"
    rff_n_features: int = 256
    subsamples: int = 1000
    sigma: float = 0.2
    eta: float = 0.1
    la: float = 1e-6
    gamma: float = 0.99
    epochs: int = 100
    warmup_episodes: int = 1
    train_episodes: int = 1
    eval_episodes: int = 1
    iter_pmd: int = 1
    parallel_envs: int = 3
    seed: int = 0
    
    def to_args(self) -> List[str]:
        """Convert config to command line arguments."""
        return [
            "--env", self.env,
            "--kernel-method", self.kernel_method,
            "--rff-n-features", str(self.rff_n_features),
            "--subsamples", str(self.subsamples),
            "--sigma", str(self.sigma),
            "--eta", str(self.eta),
            "--la", str(self.la),
            "--gamma", str(self.gamma),
            "--epochs", str(self.epochs),
            "--warmup-episodes", str(self.warmup_episodes),
            "--train-episodes", str(self.train_episodes),
            "--eval-episodes", str(self.eval_episodes),
            "--iter-pmd", str(self.iter_pmd),
            "--parallel-envs", str(self.parallel_envs),
            "--seed", str(self.seed),
            "--offline",  # Run without wandb
        ]


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps."""
    name: str
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]]
    n_seeds: int = 3
    
    def generate_configs(self) -> List[Tuple[Dict[str, Any], ExperimentConfig]]:
        """Generate all configurations for the sweep."""
        configs = []
        
        # Generate all combinations of sweep parameters
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            
            for seed in range(self.n_seeds):
                # Create config with sweep parameters
                config = ExperimentConfig(**asdict(self.base_config))
                for name, value in param_dict.items():
                    setattr(config, name, value)
                config.seed = seed
                
                configs.append((param_dict, config))
                
        return configs


# ============================================================================
# Predefined Experiment Configurations
# ============================================================================

def get_rff_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for RFF feature count sweep."""
    base = ExperimentConfig(env=env, kernel_method="rff", epochs=50)
    return SweepConfig(
        name="rff_sweep",
        base_config=base,
        sweep_params={
            "rff_n_features": [64, 128, 256, 512, 1024, 2048],
        },
        n_seeds=3,
    )


def get_nystrom_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for Nystrom subsample count sweep."""
    base = ExperimentConfig(env=env, kernel_method="exact", epochs=50)
    return SweepConfig(
        name="nystrom_sweep",
        base_config=base,
        sweep_params={
            "subsamples": [100, 500, 1000, 2000, 5000, 10000],
        },
        n_seeds=3,
    )


def get_kernel_comparison_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for kernel comparison experiments."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="kernel_comparison",
        base_config=base,
        sweep_params={
            "kernel_method": ["exact", "rff", "laplace", "matern32", "matern52"],
        },
        n_seeds=5,
    )


def get_sigma_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for bandwidth/sigma sweep."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="sigma_sweep",
        base_config=base,
        sweep_params={
            "sigma": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        },
        n_seeds=3,
    )


def get_eta_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for learning rate sweep."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="eta_sweep",
        base_config=base,
        sweep_params={
            "eta": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        n_seeds=3,
    )


def get_lambda_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for regularization sweep."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="lambda_sweep",
        base_config=base,
        sweep_params={
            "la": [1e-8, 1e-6, 1e-4, 1e-2],
        },
        n_seeds=3,
    )


def get_pmd_iterations_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for PMD iterations sweep."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="pmd_iterations_sweep",
        base_config=base,
        sweep_params={
            "iter_pmd": [1, 5, 10, 20, 50],
        },
        n_seeds=3,
    )


def get_exploration_sweep_config(env: str = "MountainCar-v0") -> SweepConfig:
    """Get configuration for exploration-exploitation sweep."""
    base = ExperimentConfig(env=env, epochs=50)
    return SweepConfig(
        name="exploration_sweep",
        base_config=base,
        sweep_params={
            "warmup_episodes": [1, 5, 10, 20],
            "train_episodes": [1, 3, 5, 10],
        },
        n_seeds=3,
    )


EXPERIMENT_CONFIGS = {
    "rff_sweep": get_rff_sweep_config,
    "nystrom_sweep": get_nystrom_sweep_config,
    "kernel_comparison": get_kernel_comparison_config,
    "sigma_sweep": get_sigma_sweep_config,
    "eta_sweep": get_eta_sweep_config,
    "lambda_sweep": get_lambda_sweep_config,
    "pmd_iterations_sweep": get_pmd_iterations_sweep_config,
    "exploration_sweep": get_exploration_sweep_config,
}


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    config: ExperimentConfig
    sweep_params: Dict[str, Any]
    final_reward: float = 0.0
    best_reward: float = 0.0
    rewards_history: List[float] = field(default_factory=list)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    total_timesteps: int = 0
    training_time: float = 0.0
    success: bool = True
    error_message: str = ""


class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        
    def run_single_experiment(
        self, 
        config: ExperimentConfig,
        sweep_params: Dict[str, Any],
        verbose: bool = True,
    ) -> ExperimentResult:
        """Run a single experiment configuration."""
        result = ExperimentResult(
            config=config,
            sweep_params=sweep_params,
        )
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = [sys.executable, "train.py"] + config.to_args()
            
            if verbose:
                print(f"\nRunning: {' '.join(cmd[:10])}...")
                print(f"  Params: {sweep_params}")
            
            # Run experiment
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if proc.returncode != 0:
                result.success = False
                result.error_message = proc.stderr[:500]
                if verbose:
                    print(f"  ERROR: {result.error_message[:100]}")
            else:
                # Parse results from output
                result = self._parse_output(proc.stdout, result)
                if verbose:
                    print(f"  Final reward: {result.final_reward:.2f}")
                    
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = "Timeout"
            if verbose:
                print("  TIMEOUT")
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if verbose:
                print(f"  EXCEPTION: {e}")
                
        result.training_time = time.time() - start_time
        self.results.append(result)
        
        return result
    
    def _parse_output(self, output: str, result: ExperimentResult) -> ExperimentResult:
        """Parse experiment output to extract metrics."""
        lines = output.split('\n')
        
        rewards = []
        for line in lines:
            # Look for reward values in output
            if "Eval reward" in line or "eval reward" in line.lower():
                try:
                    # Extract number from line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "reward" in part.lower() and i + 1 < len(parts):
                            try:
                                reward = float(parts[i + 1].strip(','))
                                rewards.append(reward)
                            except:
                                pass
                except:
                    pass
        
        if rewards:
            result.rewards_history = rewards
            result.final_reward = rewards[-1]
            result.best_reward = max(rewards)
            
        return result
    
    def run_sweep(self, sweep_config: SweepConfig, verbose: bool = True) -> List[ExperimentResult]:
        """Run all experiments in a sweep."""
        configs = sweep_config.generate_configs()
        
        print(f"\n{'='*60}")
        print(f"Running sweep: {sweep_config.name}")
        print(f"Total configurations: {len(configs)}")
        print(f"{'='*60}")
        
        for i, (sweep_params, config) in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}]", end="")
            self.run_single_experiment(config, sweep_params, verbose)
            
        # Save results
        self.save_results(sweep_config.name)
        
        return self.results
    
    def save_results(self, name: str):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{name}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
            
        print(f"\nResults saved to: {filename}")
        
        # Also save summary as JSON
        summary = self._create_summary()
        json_filename = self.output_dir / f"{name}_{timestamp}_summary.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _create_summary(self) -> Dict:
        """Create summary of results."""
        summary = {
            "n_experiments": len(self.results),
            "n_successful": sum(1 for r in self.results if r.success),
            "results": []
        }
        
        for r in self.results:
            summary["results"].append({
                "sweep_params": r.sweep_params,
                "seed": r.config.seed,
                "final_reward": r.final_reward,
                "best_reward": r.best_reward,
                "training_time": r.training_time,
                "success": r.success,
            })
            
        return summary


# ============================================================================
# Plotting Utilities
# ============================================================================

class Plotter:
    """Plotting utilities for benchmark results."""
    
    def __init__(self, output_dir: str = "benchmark_results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 12
    
    def plot_sweep_results(
        self, 
        results: List[ExperimentResult],
        sweep_param: str,
        metric: str = "final_reward",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """Plot results of a parameter sweep."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot")
            return
            
        # Group results by sweep parameter value
        grouped = {}
        for r in results:
            if not r.success:
                continue
            value = r.sweep_params.get(sweep_param)
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(getattr(r, metric, r.final_reward))
            
        # Sort by parameter value
        sorted_items = sorted(grouped.items(), key=lambda x: x[0])
        x_values = [item[0] for item in sorted_items]
        means = [np.mean(item[1]) for item in sorted_items]
        stds = [np.std(item[1]) for item in sorted_items]
        
        # Create plot
        fig, ax = plt.subplots()
        ax.errorbar(x_values, means, yerr=stds, marker='o', capsize=5, linewidth=2)
        
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} vs {sweep_param}")
        
        # Use log scale for certain parameters
        if sweep_param in ['rff_n_features', 'subsamples', 'la']:
            ax.set_xscale('log')
            
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        plt.close()
        
    def plot_learning_curves(
        self,
        results: List[ExperimentResult],
        group_by: str,
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """Plot learning curves grouped by a parameter."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot")
            return
            
        # Group results
        grouped = {}
        for r in results:
            if not r.success or not r.rewards_history:
                continue
            value = r.sweep_params.get(group_by, "default")
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(r.rewards_history)
            
        # Create plot
        fig, ax = plt.subplots()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
        
        for (value, histories), color in zip(sorted(grouped.items()), colors):
            # Pad histories to same length
            max_len = max(len(h) for h in histories)
            padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
            
            mean = np.mean(padded, axis=0)
            std = np.std(padded, axis=0)
            
            epochs = np.arange(len(mean))
            ax.plot(epochs, mean, label=f"{group_by}={value}", color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
            
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        ax.set_title(title or "Learning Curves")
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
            
        plt.close()
        
    def plot_comparison_bar(
        self,
        results: List[ExperimentResult],
        group_by: str,
        metric: str = "final_reward",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """Create bar chart comparison."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot")
            return
            
        # Group results
        grouped = {}
        for r in results:
            if not r.success:
                continue
            value = str(r.sweep_params.get(group_by, "default"))
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(getattr(r, metric, r.final_reward))
            
        # Sort and prepare data
        labels = sorted(grouped.keys())
        means = [np.mean(grouped[l]) for l in labels]
        stds = [np.std(grouped[l]) for l in labels]
        
        # Create plot
        fig, ax = plt.subplots()
        
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel(group_by)
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} by {group_by}")
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
            
        plt.close()


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_results(results: List[ExperimentResult], sweep_param: str) -> Dict:
    """Analyze sweep results and compute statistics."""
    grouped = {}
    
    for r in results:
        if not r.success:
            continue
        value = r.sweep_params.get(sweep_param)
        if value not in grouped:
            grouped[value] = {
                'final_rewards': [],
                'best_rewards': [],
                'training_times': [],
            }
        grouped[value]['final_rewards'].append(r.final_reward)
        grouped[value]['best_rewards'].append(r.best_reward)
        grouped[value]['training_times'].append(r.training_time)
        
    analysis = {}
    for value, data in grouped.items():
        analysis[value] = {
            'mean_final_reward': np.mean(data['final_rewards']),
            'std_final_reward': np.std(data['final_rewards']),
            'mean_best_reward': np.mean(data['best_rewards']),
            'std_best_reward': np.std(data['best_rewards']),
            'mean_training_time': np.mean(data['training_times']),
            'n_runs': len(data['final_rewards']),
        }
        
    return analysis


def print_analysis_table(analysis: Dict, sweep_param: str):
    """Print analysis results as a table."""
    print(f"\n{'='*80}")
    print(f"Analysis: {sweep_param}")
    print(f"{'='*80}")
    print(f"{'Value':>15} | {'Mean Reward':>12} | {'Std':>10} | {'Best':>12} | {'Time (s)':>10} | {'N':>4}")
    print(f"{'-'*80}")
    
    for value in sorted(analysis.keys()):
        data = analysis[value]
        print(f"{str(value):>15} | {data['mean_final_reward']:>12.2f} | "
              f"{data['std_final_reward']:>10.2f} | {data['mean_best_reward']:>12.2f} | "
              f"{data['mean_training_time']:>10.1f} | {data['n_runs']:>4}")
              
    print(f"{'='*80}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="POWR Benchmark Runner")
    
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()) + ["all"],
        default="rff_sweep",
        help="Experiment to run",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCar-v0",
        help="Environment to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of seeds per configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Analyze results from file instead of running",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after running",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Analyze existing results
    if args.analyze:
        with open(args.analyze, 'rb') as f:
            results = pickle.load(f)
        
        # Determine sweep parameter
        if results:
            sweep_params = list(results[0].sweep_params.keys())
            for param in sweep_params:
                analysis = analyze_results(results, param)
                print_analysis_table(analysis, param)
                
                if args.plot:
                    plotter = Plotter(args.output_dir + "/plots")
                    plotter.plot_sweep_results(
                        results, param,
                        filename=f"{param}_sweep.png"
                    )
        return
    
    # Run experiments
    experiments_to_run = (
        list(EXPERIMENT_CONFIGS.keys()) 
        if args.experiment == "all" 
        else [args.experiment]
    )
    
    for exp_name in experiments_to_run:
        config_fn = EXPERIMENT_CONFIGS[exp_name]
        sweep_config = config_fn(args.env)
        sweep_config.n_seeds = args.n_seeds
        
        if args.dry_run:
            configs = sweep_config.generate_configs()
            print(f"\nExperiment: {exp_name}")
            print(f"Would run {len(configs)} configurations:")
            for i, (params, cfg) in enumerate(configs[:5]):
                print(f"  {i+1}. {params}")
            if len(configs) > 5:
                print(f"  ... and {len(configs) - 5} more")
            continue
            
        runner = ExperimentRunner(args.output_dir)
        results = runner.run_sweep(sweep_config)
        
        # Analyze and print
        for param in sweep_config.sweep_params.keys():
            analysis = analyze_results(results, param)
            print_analysis_table(analysis, param)
            
        # Generate plots
        if args.plot:
            plotter = Plotter(args.output_dir + "/plots")
            for param in sweep_config.sweep_params.keys():
                plotter.plot_sweep_results(
                    results, param,
                    filename=f"{exp_name}_{param}.png"
                )
                plotter.plot_learning_curves(
                    results, param,
                    filename=f"{exp_name}_{param}_curves.png"
                )


if __name__ == "__main__":
    main()
