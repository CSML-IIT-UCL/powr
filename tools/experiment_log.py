#!/usr/bin/env python3
"""
Simple experiment logging for iterative tuning.
"""

import os
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = "experiment_results.json"

def load_results():
    """Load existing results."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "best": None}

def save_results(results):
    """Save results."""
    with open(LOG_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def log_experiment(config: dict, final_reward: float, notes: str = ""):
    """Log an experiment result."""
    results = load_results()
    
    entry = {
        "id": len(results["experiments"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "final_reward": final_reward,
        "notes": notes,
    }
    
    results["experiments"].append(entry)
    
    # Update best if this is better
    if results["best"] is None or final_reward > results["best"]["final_reward"]:
        results["best"] = entry
        
    save_results(results)
    return entry

def get_best():
    """Get best experiment so far."""
    results = load_results()
    return results.get("best")

def print_summary():
    """Print summary of all experiments."""
    results = load_results()
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for exp in results["experiments"]:
        config = exp["config"]
        print(f"\n#{exp['id']}: Reward={exp['final_reward']:.1f}")
        print(f"   Config: kernel={config.get('kernel_method', 'exact')}, "
              f"features={config.get('rff_n_features', 'N/A')}, "
              f"eta={config.get('eta', 0.1)}, la={config.get('la', 1e-6)}")
        if exp.get("notes"):
            print(f"   Notes: {exp['notes']}")
    
    if results["best"]:
        print(f"\n{'='*70}")
        print(f"BEST: #{results['best']['id']} with reward {results['best']['final_reward']:.1f}")
        print(f"Config: {results['best']['config']}")
    
    return results

if __name__ == "__main__":
    print_summary()
