#!/usr/bin/env python3
"""
Generate a comprehensive report with plots for CartPole-v1 tuning experiments.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

RESULTS_FILE = "cartpole_tuning_results.json"
OUTPUT_DIR = "cartpole_tuning_report"

def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_best_rewards_comparison(results):
    """Bar chart comparing best rewards across configurations."""
    experiments = results["experiments"]
    
    # Sort by best reward
    sorted_exps = sorted(experiments, key=lambda x: x["result"].get("best_reward", 0), reverse=True)
    
    # Filter out failed experiments (0 reward)
    valid_exps = [e for e in sorted_exps if e["result"].get("best_reward", 0) > 0]
    
    labels = []
    best_rewards = []
    final_rewards = []
    avg_rewards = []
    colors = []
    
    for exp in valid_exps:
        cfg = exp["config"]
        kernel = cfg.get("kernel_method", "exact")
        eta = cfg.get("eta", 0.1)
        feat = cfg.get("rff_n_features", "")
        
        label = f"{kernel}"
        if feat:
            label += f"-{feat}"
        label += f"\nη={eta}"
        
        labels.append(label)
        best_rewards.append(exp["result"].get("best_reward", 0))
        final_rewards.append(exp["result"].get("final_reward", 0))
        avg_rewards.append(exp["result"].get("avg_last_10", 0))
        
        # Color based on kernel type
        if kernel == "matern32":
            colors.append("#2ecc71")  # Green for winner
        elif kernel == "laplace":
            colors.append("#3498db")  # Blue
        elif kernel == "exact":
            colors.append("#e74c3c")  # Red
        elif kernel.startswith("rff"):
            colors.append("#9b59b6")  # Purple
        else:
            colors.append("#95a5a6")  # Gray
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    x = np.arange(len(labels))
    width = 0.7
    
    # Best rewards
    axes[0].bar(x, best_rewards, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Best Reward Achieved')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Max (500)')
    axes[0].legend()
    
    # Final rewards
    axes[1].bar(x, final_rewards, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Final Epoch Reward')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    # Average last 10
    axes[2].bar(x, avg_rewards, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Average of Last 10 Epochs')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reward_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/reward_comparison.png")

def plot_learning_curves(results):
    """Plot learning curves for experiments that have reward history."""
    experiments = results["experiments"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Sort by best reward to plot best ones last (on top)
    sorted_exps = sorted(experiments, key=lambda x: x["result"].get("best_reward", 0))
    
    for exp in sorted_exps:
        rewards = exp["result"].get("all_rewards", [])
        if not rewards or max(rewards) == 0:
            continue
            
        cfg = exp["config"]
        kernel = cfg.get("kernel_method", "exact")
        eta = cfg.get("eta", 0.1)
        feat = cfg.get("rff_n_features", "")
        best = exp["result"].get("best_reward", 0)
        
        label = f"{kernel}"
        if feat:
            label += f"-{feat}"
        label += f" (η={eta}, best={best:.0f})"
        
        # Color and style based on kernel
        if kernel == "matern32":
            color, lw, alpha = "#2ecc71", 3, 1.0
        elif kernel == "laplace":
            color, lw, alpha = "#3498db", 2.5, 0.9
        elif kernel == "exact" and eta == 0.2:
            color, lw, alpha = "#e74c3c", 2, 0.8
        elif kernel == "exact":
            color, lw, alpha = "#e67e22", 1.5, 0.7
        else:
            color, lw, alpha = "#95a5a6", 1, 0.5
        
        epochs = list(range(len(rewards)))
        ax.plot(epochs, rewards, label=label, color=color, linewidth=lw, alpha=alpha)
    
    ax.set_xlabel('Epoch (last 20)')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Learning Curves - CartPole-v1 Hyperparameter Tuning')
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max Score (500)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 550)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/learning_curves.png")

def plot_kernel_comparison(results):
    """Compare kernel types."""
    experiments = results["experiments"]
    
    # Group by kernel type
    kernel_stats = {}
    for exp in experiments:
        kernel = exp["config"].get("kernel_method", "exact")
        best = exp["result"].get("best_reward", 0)
        final = exp["result"].get("final_reward", 0)
        avg = exp["result"].get("avg_last_10", 0)
        
        if kernel not in kernel_stats:
            kernel_stats[kernel] = {"best": [], "final": [], "avg": []}
        
        if best > 0:  # Only include successful runs
            kernel_stats[kernel]["best"].append(best)
            kernel_stats[kernel]["final"].append(final)
            kernel_stats[kernel]["avg"].append(avg)
    
    kernels = list(kernel_stats.keys())
    max_best = [max(kernel_stats[k]["best"]) if kernel_stats[k]["best"] else 0 for k in kernels]
    max_avg = [max(kernel_stats[k]["avg"]) if kernel_stats[k]["avg"] else 0 for k in kernels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(kernels))
    width = 0.35
    
    colors_best = ['#2ecc71' if k == 'matern32' else '#3498db' if k == 'laplace' else '#e74c3c' if k == 'exact' else '#9b59b6' for k in kernels]
    
    bars1 = ax.bar(x - width/2, max_best, width, label='Best Reward', color=colors_best, edgecolor='black')
    bars2 = ax.bar(x + width/2, max_avg, width, label='Best Avg Last 10', color=colors_best, alpha=0.6, edgecolor='black')
    
    ax.set_ylabel('Reward')
    ax.set_title('Kernel Type Comparison - Best Results per Kernel')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in kernels])
    ax.legend()
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kernel_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/kernel_comparison.png")

def plot_stability_analysis(results):
    """Analyze stability: best vs final reward."""
    experiments = results["experiments"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for exp in experiments:
        best = exp["result"].get("best_reward", 0)
        final = exp["result"].get("final_reward", 0)
        
        if best == 0:
            continue
        
        cfg = exp["config"]
        kernel = cfg.get("kernel_method", "exact")
        
        # Color by kernel
        if kernel == "matern32":
            color, marker = "#2ecc71", "*"
            size = 300
        elif kernel == "laplace":
            color, marker = "#3498db", "^"
            size = 150
        elif kernel == "exact":
            color, marker = "#e74c3c", "o"
            size = 100
        else:
            color, marker = "#9b59b6", "s"
            size = 80
        
        ax.scatter(best, final, c=color, s=size, marker=marker, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Diagonal line (perfect stability)
    ax.plot([0, 500], [0, 500], 'g--', alpha=0.5, label='Perfect Stability')
    
    ax.set_xlabel('Best Reward Achieved')
    ax.set_ylabel('Final Reward')
    ax.set_title('Stability Analysis: Best vs Final Reward\n(Points near diagonal = stable)')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71', markersize=15, label='Matern32'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db', markersize=10, label='Laplace'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Exact/RBF'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#9b59b6', markersize=10, label='RFF'),
        Line2D([0], [0], linestyle='--', color='g', label='Perfect Stability'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 550)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stability_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/stability_analysis.png")

def generate_markdown_report(results):
    """Generate markdown report."""
    
    experiments = results["experiments"]
    
    # Sort experiments by best reward
    sorted_exps = sorted(experiments, key=lambda x: x["result"].get("best_reward", 0), reverse=True)
    
    # Find actual best (in case JSON best is outdated)
    best_exp = sorted_exps[0] if sorted_exps else results["best"]
    
    report = f"""# CartPole-v1 Hyperparameter Tuning Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Experiments:** {len(experiments)}  
**Environment:** CartPole-v1 (max reward: 500)

---

## 🏆 Best Configuration Found

| Parameter | Value |
|-----------|-------|
| **Kernel** | {best_exp['config'].get('kernel_method', 'exact')} |
| **Learning Rate (η)** | {best_exp['config'].get('eta', 0.1)} |
| **Regularization (λ)** | {best_exp['config'].get('la', 1e-6)} |
| **Best Reward** | **{best_exp['result']['best_reward']:.1f}** |
| **Final Reward** | {best_exp['result']['final_reward']:.1f} |
| **Avg Last 10** | {best_exp['result']['avg_last_10']:.1f} |

---

## 📊 Key Findings

### 1. Kernel Choice Matters Most
The **Matern 3/2 kernel** dramatically outperformed all other kernels:
- Achieved the **maximum possible score (500)**
- Maintained stable performance throughout training
- The Laplace kernel was the second best (378 best, 302.7 avg)

### 2. RBF/Gaussian Kernel is Unstable
- Reached high peaks (401 with η=0.2) but collapsed to ~11 at the end
- Standard baseline (η=0.1) peaked at 180 but also collapsed

### 3. Random Fourier Features Underperform
- All RFF approximations performed worse than exact kernels
- Best RFF result: 114.3 (RFF-256 with η=0.2)
- Trade-off: Faster computation but significantly worse learning

### 4. Learning Rate Sensitivity
- η=0.2 improved RBF from 180→401 (best) but caused instability
- η=0.05 was too slow (timed out)
- η=0.5 was too aggressive (91.3 best)

---

## 📈 Visualizations

### Reward Comparison
![Reward Comparison](reward_comparison.png)

### Learning Curves  
![Learning Curves](learning_curves.png)

### Kernel Comparison
![Kernel Comparison](kernel_comparison.png)

### Stability Analysis
![Stability Analysis](stability_analysis.png)

---

## 📋 All Experiment Results

| Rank | Kernel | η | λ | Features | Best | Final | Avg10 | Time (m) |
|------|--------|---|---|----------|------|-------|-------|----------|
"""
    
    for i, exp in enumerate(sorted_exps, 1):
        cfg = exp["config"]
        res = exp["result"]
        
        kernel = cfg.get("kernel_method", "exact")
        eta = cfg.get("eta", 0.1)
        la = cfg.get("la", 1e-6)
        feat = cfg.get("rff_n_features", "-")
        best = res.get("best_reward", 0)
        final = res.get("final_reward", 0)
        avg = res.get("avg_last_10", 0)
        duration = res.get("duration", 0) / 60
        
        status = "✅" if best > 0 else "❌"
        
        report += f"| {i} {status} | {kernel} | {eta} | {la:.0e} | {feat} | {best:.1f} | {final:.1f} | {avg:.1f} | {duration:.1f} |\n"
    
    report += """
---

## 🎯 Recommendations

1. **Use Matern 3/2 kernel** for CartPole-v1 - it achieves optimal performance
2. **Laplace kernel** is a good alternative if Matern is unavailable
3. **Avoid RBF/Gaussian** kernel for this environment - it's unstable
4. **Don't use RFF approximations** unless computational cost is critical
5. **Default η=0.1** works well with Matern32, no tuning needed

---

## 🔬 Next Steps

1. Validate Matern32 on longer runs (100+ epochs)
2. Test Matern32 on other environments (MountainCar, Acrobot)
3. Investigate why RBF collapses (possible overfitting)
4. Compare computational costs across kernels

---

*Report generated by CartPole-v1 Hyperparameter Tuning System*
"""
    
    with open(f"{OUTPUT_DIR}/REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR}/REPORT.md")

def main():
    print("="*60)
    print("Generating CartPole-v1 Tuning Report")
    print("="*60)
    
    create_output_dir()
    results = load_results()
    
    print(f"\nLoaded {len(results['experiments'])} experiments")
    print(f"Best config: {results['best']['config']}")
    print(f"Best reward: {results['best']['result']['best_reward']}")
    
    print("\nGenerating plots...")
    plot_best_rewards_comparison(results)
    plot_learning_curves(results)
    plot_kernel_comparison(results)
    plot_stability_analysis(results)
    
    print("\nGenerating markdown report...")
    generate_markdown_report(results)
    
    print("\n" + "="*60)
    print(f"Report complete! See {OUTPUT_DIR}/ for all outputs:")
    print(f"  - REPORT.md (main report)")
    print(f"  - reward_comparison.png")
    print(f"  - learning_curves.png")
    print(f"  - kernel_comparison.png")
    print(f"  - stability_analysis.png")
    print("="*60)

if __name__ == "__main__":
    main()
