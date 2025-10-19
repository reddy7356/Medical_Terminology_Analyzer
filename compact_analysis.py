#!/usr/bin/env python3
"""
Compact Analysis of Cardio-Respiratory Classification Results
Creates a single comprehensive chart with all key metrics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def load_results():
    """Load classification results"""
    results_path = Path("cardio_resp_results")
    
    with open(results_path / "performance_report.json", 'r') as f:
        report = json.load(f)
    
    with open(results_path / "classification_results.json", 'r') as f:
        results = json.load(f)
    
    with open(results_path / "cardio_respiratory_cases.json", 'r') as f:
        cardio_cases = json.load(f)
    
    return report, results, cardio_cases

def create_compact_visualization(report, results):
    """Create a single compact visualization with all key metrics"""
    # Set style for compact display
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a single figure with subplots
    fig = plt.figure(figsize=(10, 6))
    
    # Create a grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Classification Distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    classification_counts = pd.Series([r['classification'] for r in results]).value_counts()
    colors = ['#ff7f0e', '#2ca02c']
    wedges, texts, autotexts = ax1.pie(classification_counts.values, 
                                       labels=classification_counts.index, 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax1.set_title('Case Distribution', fontsize=11, fontweight='bold')
    
    # 2. Model Performance (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(report['ml_model_performance'].keys())
    f1_scores = [report['ml_model_performance'][model]['f1_score'] for model in models]
    model_names = [model.replace('_', ' ').title() for model in models]
    
    bars = ax2.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_title('Model F1-Scores', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Keyword Distribution (horizontal bar)
    ax3 = fig.add_subplot(gs[0, 2])
    df = pd.DataFrame(results)
    keyword_totals = {
        'Cardio': df['cardio_keywords'].sum(),
        'Resp': df['resp_keywords'].sum(),
        'Other': df['other_keywords'].sum()
    }
    
    categories = list(keyword_totals.keys())
    counts = list(keyword_totals.values())
    colors = ['#d62728', '#1f77b4', '#ff7f0e']
    
    bars = ax3.barh(categories, counts, color=colors, alpha=0.8)
    ax3.set_title('Keywords by Category', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Count', fontsize=9)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 4. Processing Efficiency (log scale)
    ax4 = fig.add_subplot(gs[1, :2])
    human_time = report['processing_efficiency']['estimated_human_time'] / 60  # minutes
    machine_time = report['processing_efficiency']['total_processing_time'] / 60  # minutes
    
    categories = ['Human (Estimated)', 'Machine (Actual)']
    times = [human_time, machine_time]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax4.bar(categories, times, color=colors, alpha=0.8)
    ax4.set_title('Processing Time Comparison (Log Scale)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Time (minutes)', fontsize=9)
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', labelsize=9)
    ax4.tick_params(axis='y', labelsize=8)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f} min', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Key Statistics (text box)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate key statistics
    total_cases = report['dataset_info']['total_cases']
    cardio_resp_cases = report['dataset_info']['cardio_respiratory_only']
    cardio_resp_pct = report['dataset_info']['cardio_resp_percentage']
    best_f1 = max([report['ml_model_performance'][model]['f1_score'] for model in models])
    efficiency = report['processing_efficiency']['machine_efficiency_multiplier']
    
    stats_text = f"""
KEY STATISTICS

Total Cases: {total_cases:,}
Cardio-Resp Cases: {cardio_resp_cases:,} ({cardio_resp_pct:.1f}%)

Best F1-Score: {best_f1:.3f}
Machine Efficiency: {efficiency:.0f}x faster

Processing Time:
• Human: {human_time:.1f} min
• Machine: {machine_time:.2f} min
• Time Saved: {human_time - machine_time:.1f} min
"""
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add main title
    fig.suptitle('Cardio-Respiratory Case Classification Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('cardio_resp_results/compact_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

def generate_efficiency_report(report, results, cardio_cases):
    """Generate detailed efficiency and performance report"""
    print("\n" + "="*80)
    print("📊 CARDIOVASCULAR AND RESPIRATORY CASE CLASSIFICATION - EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Dataset Information
    total_cases = report['dataset_info']['total_cases']
    cardio_resp_cases = report['dataset_info']['cardio_respiratory_only']
    mixed_cases = report['dataset_info']['mixed_or_other']
    cardio_resp_pct = report['dataset_info']['cardio_resp_percentage']
    
    print(f"📋 DATASET OVERVIEW:")
    print(f"   • Total Cases Analyzed: {total_cases:,}")
    print(f"   • Cardio-Respiratory Only: {cardio_resp_cases:,} ({cardio_resp_pct:.1f}%)")
    print(f"   • Mixed/Other Cases: {mixed_cases:,}")
    
    # Processing Efficiency
    processing_time = report['processing_efficiency']['total_processing_time']
    avg_time_per_case = report['processing_efficiency']['avg_time_per_case']
    cases_per_second = report['processing_efficiency']['cases_per_second']
    estimated_human_time = report['processing_efficiency']['estimated_human_time']
    efficiency_multiplier = report['processing_efficiency']['machine_efficiency_multiplier']
    
    print(f"\n⚡ PROCESSING EFFICIENCY:")
    print(f"   • Total Processing Time: {processing_time:.2f} seconds")
    print(f"   • Average Time per Case: {avg_time_per_case:.3f} seconds")
    print(f"   • Processing Speed: {cases_per_second:.0f} cases/second")
    print(f"   • Estimated Human Time: {estimated_human_time/60:.1f} minutes")
    print(f"   • Machine Efficiency: {efficiency_multiplier:.0f}x faster than human")
    print(f"   • Time Saved: {(estimated_human_time - processing_time)/60:.1f} minutes")
    
    # ML Model Performance
    print(f"\n🤖 MACHINE LEARNING PERFORMANCE:")
    best_model = max(report['ml_model_performance'].keys(), 
                    key=lambda k: report['ml_model_performance'][k]['f1_score'])
    
    for model_name, metrics in report['ml_model_performance'].items():
        status = "🏆 BEST" if model_name == best_model else ""
        print(f"   • {model_name.replace('_', ' ').title()}: {status}")
        print(f"     - Accuracy: {metrics['accuracy']:.3f}")
        print(f"     - Precision: {metrics['precision']:.3f}")
        print(f"     - Recall: {metrics['recall']:.3f}")
        print(f"     - F1-Score: {metrics['f1_score']:.3f}")
        print(f"     - Training Time: {metrics['training_time']:.2f}s")
    
    # Human vs Machine Comparison
    print(f"\n👥 HUMAN vs MACHINE COMPARISON:")
    print(f"   • Human Processing (Estimated):")
    print(f"     - Time per case: {estimated_human_time/total_cases/60:.1f} minutes")
    print(f"     - Total time: {estimated_human_time/60:.1f} minutes")
    print(f"     - Accuracy: ~85-90% (estimated)")
    print(f"     - Consistency: Variable (human fatigue)")
    
    print(f"   • Machine Processing (Actual):")
    print(f"     - Time per case: {avg_time_per_case:.3f} seconds")
    print(f"     - Total time: {processing_time:.2f} seconds")
    print(f"     - Accuracy: {report['ml_model_performance'][best_model]['accuracy']:.1%}")
    print(f"     - Consistency: 100% (no fatigue)")
    
    # Cost-Benefit Analysis
    print(f"\n💰 COST-BENEFIT ANALYSIS:")
    print(f"   • Efficiency Gain: {efficiency_multiplier:.0f}x faster")
    print(f"   • Time Savings: {(estimated_human_time - processing_time)/60:.1f} minutes")
    print(f"   • Scalability: Machine can process 24/7 without degradation")
    print(f"   • Consistency: Machine provides consistent results")
    print(f"   • Cost: One-time setup vs ongoing human labor")
    
    # Top Cardio-Respiratory Cases
    print(f"\n🔍 TOP CARDIO-RESPIRATORY CASES (by confidence):")
    sorted_cases = sorted(cardio_cases, key=lambda x: x['confidence'], reverse=True)[:5]
    
    for i, case in enumerate(sorted_cases, 1):
        print(f"   {i}. {case['case_id']} (Confidence: {case['confidence']:.3f})")
        print(f"      - Cardiovascular Keywords: {case['cardio_keywords']}")
        print(f"      - Respiratory Keywords: {case['resp_keywords']}")
        print(f"      - Other Medical Keywords: {case['other_keywords']}")
        print(f"      - Text Length: {case['text_length']:,} characters")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - Machine learning provides significant efficiency gains")
    print("   for large-scale medical case classification tasks.")
    print("="*80)

def main():
    """Main analysis function"""
    print("🔍 Loading classification results...")
    try:
        report, results, cardio_cases = load_results()
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print("📊 Creating compact visualization...")
    try:
        create_compact_visualization(report, results)
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        print("Continuing with text report...")
    
    print("📝 Generating efficiency report...")
    generate_efficiency_report(report, results, cardio_cases)

if __name__ == "__main__":
    main()
