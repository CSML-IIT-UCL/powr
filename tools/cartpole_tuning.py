#!/usr/bin/env python3
"""
Systematic hyperparameter tuning for CartPole-v1.
Runs experiments and tracks results for comparison.
"""

import subprocess
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Results tracking
RESULTS_FILE = "cartpole_tuning_results.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "best": None}

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_experiment(config: dict, epochs: int = 50, eval_episodes: int = 5):
    """Run a single experiment and return the best eval reward achieved."""
    
    cmd = [
        "python", "train.py",
        "--env", "CartPole-v1",
        "--epochs", str(epochs),
        "--eval-episodes", str(eval_episodes),
        "--offline",
    ]
    
    # Add config parameters
    for key, value in config.items():
        if key == "kernel_method":
            cmd.extend(["--kernel-method", str(value)])
        elif key == "rff_n_features":
            cmd.extend(["--rff-n-features", str(value)])
        elif key == "eta":
            cmd.extend(["--eta", str(value)])
        elif key == "la":
            cmd.extend(["--la", str(value)])
        elif key == "sigma":
            cmd.extend(["--sigma", str(value)])
        elif key == "warmup_episodes":
            cmd.extend(["--warmup-episodes", str(value)])
        elif key == "train_episodes":
            cmd.extend(["--train-episodes", str(value)])
        elif key == "iter_pmd":
            cmd.extend(["--iter-pmd", str(value)])
        elif key == "subsamples":
            cmd.extend(["--subsamples", str(value)])
        elif key == "seed":
            cmd.extend(["--seed", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        duration = time.time() - start_time
        
        # Parse output to get best eval reward
        output = result.stdout + result.stderr
        
        # Extract eval rewards from output
        eval_rewards = []
        for line in output.split('\n'):
            if "Eval reward" in line:
                try:
                    # Parse the reward value
                    parts = line.split('|')
                    if len(parts) >= 3:
                        reward_str = parts[2].strip().rstrip('|').strip()
                        eval_rewards.append(float(reward_str))
                except:
                    pass
        
        best_reward = max(eval_rewards) if eval_rewards else 0
        final_reward = eval_rewards[-1] if eval_rewards else 0
        avg_last_10 = sum(eval_rewards[-10:]) / len(eval_rewards[-10:]) if len(eval_rewards) >= 10 else final_reward
        
        return {
            "success": result.returncode == 0,
            "best_reward": best_reward,
            "final_reward": final_reward,
            "avg_last_10": avg_last_10,
            "duration": duration,
            "all_rewards": eval_rewards[-20:],  # Keep last 20 for analysis
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "duration": 3600}
    except Exception as e:
        return {"success": False, "error": str(e), "duration": time.time() - start_time}

def log_experiment(config, result):
    """Log experiment result."""
    results = load_results()
    
    entry = {
        "id": len(results["experiments"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "result": result,
    }
    
    results["experiments"].append(entry)
    
    # Update best
    if result.get("success") and result.get("best_reward", 0) > 0:
        if results["best"] is None or result["best_reward"] > results["best"]["result"]["best_reward"]:
            results["best"] = entry
    
    save_results(results)
    return entry

def print_summary():
    """Print summary of all experiments."""
    results = load_results()
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY - CartPole-v1 Hyperparameter Tuning")
    print("="*80)
    
    # Sort by best reward
    sorted_exps = sorted(
        results["experiments"],
        key=lambda x: x["result"].get("best_reward", 0),
        reverse=True
    )
    
    print(f"\n{'ID':<4} {'Best':<8} {'Final':<8} {'Avg10':<8} {'Time':<8} Config")
    print("-"*80)
    
    for exp in sorted_exps:
        res = exp["result"]
        cfg = exp["config"]
        
        config_str = f"kernel={cfg.get('kernel_method', 'exact')}"
        if cfg.get('rff_n_features'):
            config_str += f", feat={cfg['rff_n_features']}"
        config_str += f", eta={cfg.get('eta', 0.1)}, la={cfg.get('la', 1e-6)}"
        
        print(f"{exp['id']:<4} {res.get('best_reward', 0):<8.1f} {res.get('final_reward', 0):<8.1f} "
              f"{res.get('avg_last_10', 0):<8.1f} {res.get('duration', 0)/60:<8.1f}m {config_str}")
    
    if results["best"]:
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION: #{results['best']['id']}")
        print(f"Best Reward: {results['best']['result']['best_reward']:.1f}")
        print(f"Config: {results['best']['config']}")
        print("="*80)

def main():
    """Run systematic hyperparameter tuning."""
    
    experiments = [
        # Baseline configurations
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6},
        
        # Learning rate variations
        {"kernel_method": "exact", "eta": 0.05, "la": 1e-6},
        {"kernel_method": "exact", "eta": 0.2, "la": 1e-6},
        {"kernel_method": "exact", "eta": 0.5, "la": 1e-6},
        
        # Regularization variations
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-4},
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-8},
        
        # RFF approximation with different feature counts
        {"kernel_method": "rff", "rff_n_features": 128, "eta": 0.1, "la": 1e-6},
        {"kernel_method": "rff", "rff_n_features": 256, "eta": 0.1, "la": 1e-6},
        {"kernel_method": "rff", "rff_n_features": 512, "eta": 0.1, "la": 1e-6},
        
        # RFF with tuned eta
        {"kernel_method": "rff", "rff_n_features": 256, "eta": 0.2, "la": 1e-6},
        {"kernel_method": "rff", "rff_n_features": 256, "eta": 0.5, "la": 1e-6},
        
        # Different kernels
        {"kernel_method": "laplace", "eta": 0.1, "la": 1e-6},
        {"kernel_method": "matern32", "eta": 0.1, "la": 1e-6},
        {"kernel_method": "matern52", "eta": 0.1, "la": 1e-6},
        
        # Exploration settings
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "warmup_episodes": 5},
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "train_episodes": 5},
        
        # PMD iterations
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "iter_pmd": 3},
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "iter_pmd": 5},
        
        # Subsampling (Nystrom)
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "subsamples": 500},
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "subsamples": 1000},
        
        # Sigma variations
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "sigma": 0.5},
        {"kernel_method": "exact", "eta": 0.1, "la": 1e-6, "sigma": 2.0},
    ]
    
    print(f"\nStarting CartPole-v1 Hyperparameter Tuning")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time: {len(experiments) * 15} - {len(experiments) * 30} minutes")
    
    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running experiment...")
        
        result = run_experiment(config, epochs=50, eval_episodes=3)
        entry = log_experiment(config, result)
        
        print(f"\nResult: best={result.get('best_reward', 'N/A')}, "
              f"final={result.get('final_reward', 'N/A')}, "
              f"time={result.get('duration', 0)/60:.1f}m")
        
        # Print intermediate summary
        if (i + 1) % 5 == 0:
            print_summary()
    
    # Final summary
    print_summary()
    
    print("\n\nTuning complete! Check cartpole_tuning_results.json for full results.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary()
    else:
        main()
