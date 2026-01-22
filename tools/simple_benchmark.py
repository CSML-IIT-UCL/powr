#!/usr/bin/env python3
"""
Simple benchmark script to compare exact vs RFF approximation.

This produces plots similar to your friend's:
1. Approximation Error vs Number of Features
2. Wall-time vs Number of Features
3. Eval Reward vs Timesteps

Usage:
    python tools/simple_benchmark.py --env MountainCar-v0
    python tools/simple_benchmark.py --env MountainCar-v0 --quick
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax.numpy as jnp
import gymnasium as gym

try:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    HAS_MATPLOTLIB = True
except:
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except:
        HAS_MATPLOTLIB = False
        print("Warning: matplotlib not available")

from powr.kernels import gaussian_kernel, gaussian_kernel_diag, RFFGaussian


def get_sigma_for_env(env_name):
    """Get appropriate sigma for environment."""
    if env_name == "MountainCar-v0":
        return [0.1, 0.01]
    elif env_name == "CartPole-v1":
        return 0.2
    elif env_name == "LunarLander-v2":
        return [0.2] * 6 + [0.0001, 0.0001]
    else:
        return 0.2


def benchmark_kernel_approximation(env_name, n_features_list, n_samples=1000, n_trials=5):
    """
    Benchmark RFF approximation quality against exact kernel.
    
    Returns data for plotting:
    - Approximation error vs features
    - Wall-time vs features
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking Kernel Approximation: {env_name}")
    print(f"Features: {n_features_list}")
    print(f"Samples: {n_samples}, Trials: {n_trials}")
    print(f"{'='*60}")
    
    # Create environment and get sample data
    env = gym.make(env_name)
    sigma = get_sigma_for_env(env_name)
    
    # Collect sample states
    states = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    env.close()
    
    X = jnp.array(states)
    print(f"Collected {len(X)} samples, shape: {X.shape}")
    
    # Create exact kernel
    if isinstance(sigma, list):
        exact_kernel = gaussian_kernel_diag(sigma)
    else:
        exact_kernel = gaussian_kernel(sigma, method="exact")
    
    # Compute exact kernel matrix (ground truth)
    print("Computing exact kernel matrix...")
    K_exact = np.array(exact_kernel(X, X))
    
    results = {
        'n_features': [],
        'approx_error_mean': [],
        'approx_error_std': [],
        'relative_error_mean': [],
        'relative_error_std': [],
        'time_rff_mean': [],
        'time_rff_std': [],
        'time_exact': 0,
    }
    
    # Time exact kernel
    start = time.time()
    for _ in range(n_trials):
        _ = exact_kernel(X, X)
    results['time_exact'] = (time.time() - start) / n_trials
    print(f"Exact kernel time: {results['time_exact']:.4f}s")
    
    # Test each feature count
    for n_features in n_features_list:
        print(f"\nTesting n_features={n_features}...")
        
        errors = []
        rel_errors = []
        times = []
        
        for trial in range(n_trials):
            # Create RFF kernel with different seed each trial
            rff_kernel = RFFGaussian(sigma=sigma, n_features=n_features, seed=trial)
            
            # Time and compute RFF kernel
            start = time.time()
            K_rff = np.array(rff_kernel(X, X))
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Compute approximation error
            diff = np.abs(K_exact - K_rff)
            error = np.mean(diff)
            rel_error = np.mean(diff / (np.abs(K_exact) + 1e-10))
            
            errors.append(error)
            rel_errors.append(rel_error)
        
        results['n_features'].append(n_features)
        results['approx_error_mean'].append(np.mean(errors))
        results['approx_error_std'].append(np.std(errors))
        results['relative_error_mean'].append(np.mean(rel_errors))
        results['relative_error_std'].append(np.std(rel_errors))
        results['time_rff_mean'].append(np.mean(times))
        results['time_rff_std'].append(np.std(times))
        
        print(f"  Error: {np.mean(errors):.6f} ± {np.std(errors):.6f}")
        print(f"  Relative Error: {np.mean(rel_errors):.4%} ± {np.std(rel_errors):.4%}")
        print(f"  Time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    
    return results


def plot_results(results, env_name, output_dir):
    """Generate plots similar to your friend's."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = results['n_features']
    
    # Plot 1: Approximation Error vs Features
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        n_features, 
        results['approx_error_mean'],
        yerr=results['approx_error_std'],
        marker='o', 
        capsize=5, 
        linewidth=2,
        markersize=8,
        color='steelblue',
        label='RFF Approximation Error'
    )
    ax.set_xlabel('Number of Features (D)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Approximation Error vs Features ({env_name})', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    filepath = output_dir / f'approx_error_vs_features_{env_name}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Plot 2: Wall-time vs Features
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        n_features,
        results['time_rff_mean'],
        yerr=results['time_rff_std'],
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8,
        color='coral',
        label='RFF Kernel'
    )
    ax.axhline(
        y=results['time_exact'],
        color='seagreen',
        linestyle='--',
        linewidth=2,
        label='Exact Kernel'
    )
    ax.set_xlabel('Number of Features (D)', fontsize=12)
    ax.set_ylabel('Wall-time (seconds)', fontsize=12)
    ax.set_title(f'Wall-time vs Features ({env_name})', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    filepath = output_dir / f'walltime_vs_features_{env_name}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Plot 3: Combined plot (2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.errorbar(
        n_features,
        results['approx_error_mean'],
        yerr=results['approx_error_std'],
        marker='o',
        capsize=5,
        linewidth=2,
        color='steelblue'
    )
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Approximation Error')
    ax.set_title('Approximation Error vs Features')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.errorbar(
        n_features,
        results['time_rff_mean'],
        yerr=results['time_rff_std'],
        marker='o',
        capsize=5,
        linewidth=2,
        color='coral',
        label='RFF'
    )
    ax.axhline(y=results['time_exact'], color='seagreen', linestyle='--', label='Exact')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Wall-time (s)')
    ax.set_title('Wall-time vs Features')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'RFF Approximation Benchmark ({env_name})', fontsize=14)
    plt.tight_layout()
    
    filepath = output_dir / f'combined_benchmark_{env_name}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Features':>10} | {'Error':>12} | {'Rel. Error':>12} | {'Time (s)':>10}")
    print(f"{'-'*50}")
    for i, n in enumerate(n_features):
        print(f"{n:>10} | {results['approx_error_mean'][i]:>12.6f} | "
              f"{results['relative_error_mean'][i]:>11.2%} | {results['time_rff_mean'][i]:>10.4f}")
    print(f"{'Exact':>10} | {'0':>12} | {'0%':>12} | {results['time_exact']:>10.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RFF Benchmark")
    parser.add_argument("--env", type=str, default="MountainCar-v0")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer features")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="benchmark_results/simple")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Feature counts to test
    if args.quick:
        n_features_list = [64, 256, 1024]
    else:
        n_features_list = [32, 64, 128, 256, 512, 1024, 2048]
    
    # Run benchmark
    results = benchmark_kernel_approximation(
        args.env,
        n_features_list,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
    )
    
    # Generate plots
    plot_results(results, args.env, args.output_dir)
    
    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
