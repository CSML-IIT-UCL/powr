#!/usr/bin/env python3
"""
Run all POWR experiments.

This script runs the complete set of experiments from the plan:
1. RFF feature count sweep
2. Nystrom subsample count sweep  
3. Kernel comparison
4. Exploration-exploitation sweep
5. Oracle experiments

Usage:
    python tools/run_experiments.py --all
    python tools/run_experiments.py --experiment rff_sweep
    python tools/run_experiments.py --quick  # Quick test with fewer configs
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd: list, description: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN - not executing]")
        return True
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_rff_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run RFF feature count sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "rff_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"RFF Feature Sweep ({env})", dry_run)


def run_nystrom_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run Nystrom subsample count sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "nystrom_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"Nystrom Sweep ({env})", dry_run)


def run_kernel_comparison(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run kernel comparison experiments."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "kernel_comparison",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"Kernel Comparison ({env})", dry_run)


def run_exploration_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run exploration-exploitation sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "exploration_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"Exploration Sweep ({env})", dry_run)


def run_eta_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run learning rate sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "eta_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"Learning Rate Sweep ({env})", dry_run)


def run_lambda_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run regularization sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "lambda_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"Regularization Sweep ({env})", dry_run)


def run_pmd_iterations_sweep(env: str, n_seeds: int, dry_run: bool, quick: bool) -> bool:
    """Run PMD iterations sweep."""
    cmd = [
        sys.executable, "tools/benchmark.py",
        "--experiment", "pmd_iterations_sweep",
        "--env", env,
        "--n-seeds", str(n_seeds),
        "--plot",
    ]
    if dry_run:
        cmd.append("--dry-run")
    
    return run_command(cmd, f"PMD Iterations Sweep ({env})", dry_run)


def run_oracle_experiments(env: str, n_samples: int, dry_run: bool, quick: bool) -> bool:
    """Run oracle experiments for error decomposition."""
    if env not in ["FrozenLake-v1", "Taxi-v3"]:
        print(f"Skipping oracle experiments for {env} (requires tabular environment)")
        return True
    
    cmd = [
        sys.executable, "tools/oracle_experiments.py",
        "--env", env,
        "--n-samples", str(n_samples),
        "--sample-sweep" if not quick else "",
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    return run_command(cmd, f"Oracle Experiments ({env})", dry_run)


def run_policy_analysis(dry_run: bool) -> bool:
    """Run policy analysis demo."""
    cmd = [
        sys.executable, "tools/policy_analysis.py",
    ]
    
    return run_command(cmd, "Policy Analysis Demo", dry_run)


EXPERIMENTS = {
    'rff_sweep': run_rff_sweep,
    'nystrom_sweep': run_nystrom_sweep,
    'kernel_comparison': run_kernel_comparison,
    'exploration_sweep': run_exploration_sweep,
    'eta_sweep': run_eta_sweep,
    'lambda_sweep': run_lambda_sweep,
    'pmd_iterations_sweep': run_pmd_iterations_sweep,
    'oracle': run_oracle_experiments,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run POWR Experiments")
    
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()) + ['all', 'core'],
        default=None,
        help="Experiment to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Run core experiments (rff_sweep, nystrom_sweep, kernel_comparison)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer configurations",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCar-v0",
        help="Environment to use",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple environments to run",
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
        help="Print commands without executing",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine environments
    if args.envs:
        environments = args.envs
    else:
        environments = [args.env]
    
    # Reduce seeds for quick mode
    n_seeds = 1 if args.quick else args.n_seeds
    
    # Determine experiments to run
    if args.all:
        experiments = list(EXPERIMENTS.keys())
    elif args.core:
        experiments = ['rff_sweep', 'nystrom_sweep', 'kernel_comparison']
    elif args.experiment:
        experiments = [args.experiment]
    else:
        print("Please specify --experiment, --core, or --all")
        return
    
    # Print summary
    print(f"\n{'#'*60}")
    print(f"# POWR Experiment Runner")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Experiments: {experiments}")
    print(f"# Environments: {environments}")
    print(f"# Seeds per config: {n_seeds}")
    print(f"# Quick mode: {args.quick}")
    print(f"# Dry run: {args.dry_run}")
    print(f"{'#'*60}")
    
    # Run experiments
    results = {}
    
    for exp_name in experiments:
        for env in environments:
            exp_fn = EXPERIMENTS.get(exp_name)
            if exp_fn:
                key = f"{exp_name}_{env}"
                
                if exp_name == 'oracle':
                    success = exp_fn(env, 1000, args.dry_run, args.quick)
                else:
                    success = exp_fn(env, n_seeds, args.dry_run, args.quick)
                    
                results[key] = success
    
    # Run policy analysis separately
    if args.all or 'policy_analysis' in experiments:
        results['policy_analysis'] = run_policy_analysis(args.dry_run)
    
    # Print summary
    print(f"\n{'#'*60}")
    print(f"# Summary")
    print(f"{'#'*60}")
    
    for key, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {key}: {status}")
    
    n_success = sum(results.values())
    n_total = len(results)
    print(f"\nTotal: {n_success}/{n_total} experiments completed successfully")
    
    if not args.dry_run:
        print(f"\nResults saved to: benchmark_results/")


if __name__ == "__main__":
    main()
