#!/usr/bin/env python3
"""
Generate report from POWR experiment results.

This script compiles results from all experiments into a comprehensive report.

Usage:
    python tools/generate_report.py --results-dir benchmark_results
"""

import os
import sys
import glob
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Report Generation
# ============================================================================

class ReportGenerator:
    """Generate comprehensive report from experiment results."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.report_dir = self.results_dir / "report"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.sections = []
        
    def load_pickle_results(self, pattern: str) -> List[Dict]:
        """Load all pickle files matching pattern."""
        files = glob.glob(str(self.results_dir / pattern))
        results = []
        for f in files:
            try:
                with open(f, 'rb') as fp:
                    results.append({
                        'file': f,
                        'data': pickle.load(fp),
                    })
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return results
    
    def load_json_results(self, pattern: str) -> List[Dict]:
        """Load all JSON files matching pattern."""
        files = glob.glob(str(self.results_dir / pattern))
        results = []
        for f in files:
            try:
                with open(f, 'r') as fp:
                    results.append({
                        'file': f,
                        'data': json.load(fp),
                    })
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return results
    
    def generate_section(self, title: str, content: str):
        """Add a section to the report."""
        self.sections.append({
            'title': title,
            'content': content,
        })
        
    def analyze_sweep_results(self, results: List[Dict], sweep_param: str) -> Dict:
        """Analyze sweep experiment results."""
        analysis = {}
        
        for result in results:
            data = result['data']
            
            if isinstance(data, list):
                # List of ExperimentResult objects
                grouped = {}
                for r in data:
                    if not hasattr(r, 'success') or not r.success:
                        continue
                    value = r.sweep_params.get(sweep_param)
                    if value not in grouped:
                        grouped[value] = []
                    grouped[value].append(r.final_reward)
                    
                for value, rewards in grouped.items():
                    if value not in analysis:
                        analysis[value] = {'rewards': [], 'file': result['file']}
                    analysis[value]['rewards'].extend(rewards)
                    
            elif isinstance(data, dict):
                # Summary data
                for item in data.get('results', []):
                    value = item.get('sweep_params', {}).get(sweep_param)
                    if value is not None:
                        if value not in analysis:
                            analysis[value] = {'rewards': []}
                        analysis[value]['rewards'].append(item.get('final_reward', 0))
                        
        # Compute statistics
        for value in analysis:
            rewards = analysis[value]['rewards']
            if rewards:
                analysis[value]['mean'] = np.mean(rewards)
                analysis[value]['std'] = np.std(rewards)
                analysis[value]['n'] = len(rewards)
                
        return analysis
    
    def format_table(self, headers: List[str], rows: List[List]) -> str:
        """Format data as markdown table."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Build table
        lines = []
        
        # Header
        header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(f"| {header_line} |")
        
        # Separator
        sep_line = " | ".join("-" * widths[i] for i in range(len(headers)))
        lines.append(f"| {sep_line} |")
        
        # Rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            lines.append(f"| {row_line} |")
            
        return "\n".join(lines)
    
    def generate_rff_section(self):
        """Generate section for RFF experiments."""
        results = self.load_json_results("rff_sweep*_summary.json")
        
        if not results:
            content = "No RFF sweep results found.\n"
            content += "\nTo run: `python tools/run_experiments.py --experiment rff_sweep`"
        else:
            analysis = self.analyze_sweep_results(
                self.load_pickle_results("rff_sweep*.pkl"),
                "rff_n_features"
            )
            
            content = "### RFF Feature Count Sweep\n\n"
            content += "Effect of Random Fourier Features count on performance:\n\n"
            
            headers = ["Features", "Mean Reward", "Std", "N"]
            rows = []
            for value in sorted(analysis.keys()):
                stats = analysis[value]
                rows.append([
                    value,
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('std', 0):.2f}",
                    stats.get('n', 0),
                ])
            
            content += self.format_table(headers, rows) + "\n"
            
        self.generate_section("RFF Feature Sweep", content)
        
    def generate_nystrom_section(self):
        """Generate section for Nystrom experiments."""
        results = self.load_json_results("nystrom_sweep*_summary.json")
        
        if not results:
            content = "No Nystrom sweep results found.\n"
            content += "\nTo run: `python tools/run_experiments.py --experiment nystrom_sweep`"
        else:
            analysis = self.analyze_sweep_results(
                self.load_pickle_results("nystrom_sweep*.pkl"),
                "subsamples"
            )
            
            content = "### Nystrom Subsample Sweep\n\n"
            content += "Effect of Nystrom subsample count on performance:\n\n"
            
            headers = ["Subsamples", "Mean Reward", "Std", "N"]
            rows = []
            for value in sorted(analysis.keys()):
                stats = analysis[value]
                rows.append([
                    value,
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('std', 0):.2f}",
                    stats.get('n', 0),
                ])
            
            content += self.format_table(headers, rows) + "\n"
            
        self.generate_section("Nystrom Sweep", content)
        
    def generate_kernel_section(self):
        """Generate section for kernel comparison."""
        results = self.load_json_results("kernel_comparison*_summary.json")
        
        if not results:
            content = "No kernel comparison results found.\n"
            content += "\nTo run: `python tools/run_experiments.py --experiment kernel_comparison`"
        else:
            analysis = self.analyze_sweep_results(
                self.load_pickle_results("kernel_comparison*.pkl"),
                "kernel_method"
            )
            
            content = "### Kernel Comparison\n\n"
            content += "Comparison of different kernel types:\n\n"
            
            headers = ["Kernel", "Mean Reward", "Std", "N"]
            rows = []
            for value in sorted(analysis.keys()):
                stats = analysis[value]
                rows.append([
                    value,
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('std', 0):.2f}",
                    stats.get('n', 0),
                ])
            
            content += self.format_table(headers, rows) + "\n"
            
        self.generate_section("Kernel Comparison", content)
        
    def generate_oracle_section(self):
        """Generate section for oracle experiments."""
        oracle_dir = self.results_dir / "oracle"
        
        if not oracle_dir.exists():
            content = "No oracle experiment results found.\n"
            content += "\nTo run: `python tools/oracle_experiments.py`"
        else:
            content = "### Error Decomposition Analysis\n\n"
            content += "Oracle experiments to identify bottlenecks:\n\n"
            
            # Try to load results
            results_file = oracle_dir / "oracle_results.pkl"
            if results_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                    
                headers = ["Experiment", "Q Error (inf)", "Operator Error", "Reward Error"]
                rows = []
                
                for exp_type, result in results.items():
                    rows.append([
                        exp_type,
                        f"{result.q_error_inf:.4f}",
                        f"{result.operator_error_hs:.4f}",
                        f"{result.reward_error_inf:.4f}",
                    ])
                    
                content += self.format_table(headers, rows) + "\n"
                
                # Determine bottleneck
                baseline_q = results['baseline'].q_error_inf
                oracle_T_q = results['oracle_T'].q_error_inf
                oracle_r_q = results['oracle_r'].q_error_inf
                
                content += "\n**Bottleneck Analysis:**\n"
                if oracle_T_q < oracle_r_q:
                    content += f"- Primary bottleneck: **Transition Operator (T)**\n"
                    content += f"- Error reduced by {(1 - oracle_r_q/baseline_q)*100:.1f}% when using true T\n"
                else:
                    content += f"- Primary bottleneck: **Reward Function (r)**\n"
                    content += f"- Error reduced by {(1 - oracle_T_q/baseline_q)*100:.1f}% when using true r\n"
            else:
                content += "Results file not found.\n"
            
        self.generate_section("Error Decomposition", content)
        
    def generate_hyperparameter_section(self):
        """Generate section for hyperparameter sensitivity."""
        content = "### Hyperparameter Sensitivity\n\n"
        
        # Eta sweep
        eta_results = self.load_json_results("eta_sweep*_summary.json")
        if eta_results:
            content += "#### Learning Rate (eta)\n\n"
            analysis = self.analyze_sweep_results(
                self.load_pickle_results("eta_sweep*.pkl"),
                "eta"
            )
            headers = ["Eta", "Mean Reward", "Std"]
            rows = [[v, f"{analysis[v].get('mean', 0):.2f}", f"{analysis[v].get('std', 0):.2f}"] 
                    for v in sorted(analysis.keys())]
            content += self.format_table(headers, rows) + "\n\n"
            
        # Lambda sweep
        lambda_results = self.load_json_results("lambda_sweep*_summary.json")
        if lambda_results:
            content += "#### Regularization (lambda)\n\n"
            analysis = self.analyze_sweep_results(
                self.load_pickle_results("lambda_sweep*.pkl"),
                "la"
            )
            headers = ["Lambda", "Mean Reward", "Std"]
            rows = [[v, f"{analysis[v].get('mean', 0):.2f}", f"{analysis[v].get('std', 0):.2f}"] 
                    for v in sorted(analysis.keys())]
            content += self.format_table(headers, rows) + "\n\n"
            
        if not eta_results and not lambda_results:
            content += "No hyperparameter sensitivity results found.\n"
            content += "\nTo run: `python tools/run_experiments.py --experiment eta_sweep`\n"
            
        self.generate_section("Hyperparameter Sensitivity", content)
        
    def generate_report(self) -> str:
        """Generate the full report."""
        # Generate all sections
        self.generate_rff_section()
        self.generate_nystrom_section()
        self.generate_kernel_section()
        self.generate_oracle_section()
        self.generate_hyperparameter_section()
        
        # Build report
        report = []
        report.append("# POWR Experiment Report")
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Table of contents
        report.append("## Table of Contents")
        report.append("")
        for i, section in enumerate(self.sections, 1):
            report.append(f"{i}. [{section['title']}](#{section['title'].lower().replace(' ', '-')})")
        report.append("")
        report.append("---")
        report.append("")
        
        # Sections
        for section in self.sections:
            report.append(f"## {section['title']}")
            report.append("")
            report.append(section['content'])
            report.append("")
            report.append("---")
            report.append("")
            
        # Summary and recommendations
        report.append("## Summary and Recommendations")
        report.append("")
        report.append("Based on the experimental results:")
        report.append("")
        report.append("1. **Kernel Selection**: Review kernel comparison results to choose the best kernel for your environment.")
        report.append("2. **Approximation Quality**: RFF and Nystrom approximations trade accuracy for speed.")
        report.append("3. **Bottleneck**: Oracle experiments reveal whether to focus on improving T or r estimation.")
        report.append("4. **Hyperparameters**: Sensitivity analysis shows which parameters need careful tuning.")
        report.append("")
        report.append("### Next Steps")
        report.append("")
        report.append("- Run full experiments: `python tools/run_experiments.py --all`")
        report.append("- Analyze specific results: `python tools/benchmark.py --analyze <results.pkl>`")
        report.append("- Generate plots: `python tools/benchmark.py --experiment <name> --plot`")
        report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "experiment_report.md"):
        """Save report to file."""
        report = self.generate_report()
        
        filepath = self.report_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)
            
        print(f"Report saved to: {filepath}")
        return filepath


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate POWR Experiment Report")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_report.md",
        help="Output filename for report",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Generating POWR Experiment Report")
    print("=" * 50)
    
    generator = ReportGenerator(args.results_dir)
    filepath = generator.save_report(args.output)
    
    print(f"\nReport generated successfully!")
    print(f"View: {filepath}")


if __name__ == "__main__":
    main()
