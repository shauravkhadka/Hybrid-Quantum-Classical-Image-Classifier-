"""
Comprehensive comparison of classical vs hybrid models
Generates performance metrics and visualizations
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_results(model_name):
    """Load results for a model."""
    results_file = f'results/{model_name}_results.json'
    
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def compare_models():
    """Compare classical and hybrid models."""
    
    print("="*70)
    print("HYBRID QUANTUM-CLASSICAL CLASSIFIER - PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load results
    classical_results = load_results('classical')
    hybrid_results = load_results('hybrid')
    
    if classical_results is None or hybrid_results is None:
        print("\nError: Could not load results. Make sure to run training scripts first:")
        print("  python scripts/train_classical.py")
        print("  python scripts/train_hybrid.py")
        return
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Test Loss'],
        'Classical': [
            classical_results['test_accuracy'],
            classical_results['test_precision'],
            classical_results['test_recall'],
            classical_results['test_f1'],
            classical_results['test_loss']
        ],
        'Hybrid': [
            hybrid_results['test_accuracy'],
            hybrid_results['test_precision'],
            hybrid_results['test_recall'],
            hybrid_results['test_f1'],
            hybrid_results['test_loss']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    df['Difference'] = df['Hybrid'] - df['Classical']
    df['% Change'] = (df['Difference'] / df['Classical'].abs() * 100).round(2)
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Calculate additional metrics
    print("\n" + "="*70)
    print("ADDITIONAL METRICS")
    print("="*70)
    
    if 'inference_time_ms' in hybrid_results:
        print(f"Hybrid Inference Time: {hybrid_results['inference_time_ms']:.2f} ms/batch")
    
    print(f"\nHybrid Model Configuration:")
    if 'config' in hybrid_results:
        for key, value in hybrid_results['config'].items():
            print(f"  {key}: {value}")
    
    # Plot comparisons
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Classical vs Hybrid Model Comparison', fontsize=16, fontweight='bold')
    
    # Metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    classical_scores = [
        classical_results['test_accuracy'],
        classical_results['test_precision'],
        classical_results['test_recall'],
        classical_results['test_f1']
    ]
    hybrid_scores = [
        hybrid_results['test_accuracy'],
        hybrid_results['test_precision'],
        hybrid_results['test_recall'],
        hybrid_results['test_f1']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, classical_scores, width, label='Classical', alpha=0.8)
    axes[0, 0].bar(x + width/2, hybrid_scores, width, label='Hybrid', alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Loss comparison
    axes[0, 1].bar(['Classical', 'Hybrid'], 
                   [classical_results['test_loss'], hybrid_results['test_loss']],
                   color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Test Loss Comparison')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Training loss history
    axes[1, 0].plot(classical_results['history']['train_loss'], 
                    marker='o', label='Classical', linewidth=2)
    axes[1, 0].plot(hybrid_results['history']['train_loss'], 
                    marker='s', label='Hybrid', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss History')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy history
    axes[1, 1].plot(classical_results['history']['val_acc'], 
                    marker='o', label='Classical', linewidth=2)
    axes[1, 1].plot(hybrid_results['history']['val_acc'], 
                    marker='s', label='Hybrid', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy History')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to results/comparison.png")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    accuracy_diff = hybrid_results['test_accuracy'] - classical_results['test_accuracy']
    f1_diff = hybrid_results['test_f1'] - classical_results['test_f1']
    
    print(f"\nAccuracy Difference (Hybrid - Classical): {accuracy_diff:+.4f}")
    print(f"F1-Score Difference (Hybrid - Classical): {f1_diff:+.4f}")
    
    if accuracy_diff >= 0:
        print(f"\n✓ Hybrid model achieved comparable or better accuracy")
    else:
        print(f"\n⚠ Hybrid model has lower accuracy (expected trade-off with quantum advantage)")
    
    print("\n✓ Comparison complete!")
    print("="*70)


if __name__ == '__main__':
    compare_models()
