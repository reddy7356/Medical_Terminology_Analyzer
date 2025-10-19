#!/usr/bin/env python3
"""
Simple Analysis of Cardio-Respiratory Classification Results
Creates basic visualizations and performance summary
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

def create_simple_visualizations(report, results):
    """Create simple visualizations"""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots (smaller size)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Cardio-Respiratory Case Classification Analysis', fontsize=14, fontweight='bold')
    
    # 1. Classification Distribution
    classification_counts = pd.Series([r['classification'] for r in results]).value_counts()
    colors = ['#ff7f0e', '#2ca02c']
    wedges, texts, autotexts = ax1.pie(classification_counts.values, 
                                       labels=classification_counts.index, 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax1.set_title('Case Classification Distribution')
    
    # 2. Model Performance Comparison
    models = list(report['ml_model_performance'].keys())
    accuracies = [report['ml_model_performance'][model]['accuracy'] for model in models]
    f1_scores = [report['ml_model_performance'][model]['f1_score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax2.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('Models', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Model Performance Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([model.replace('_', ' ').title() for model in models], rotation=45, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars (smaller font)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    # 3. Keyword Distribution
    df = pd.DataFrame(results)
    keyword_totals = {
        'Cardiovascular': df['cardio_keywords'].sum(),
        'Respiratory': df['resp_keywords'].sum(),
        'Other Medical': df['other_keywords'].sum()
    }
    
    categories = list(keyword_totals.keys())
    counts = list(keyword_totals.values())
    colors = ['#d62728', '#1f77b4', '#ff7f0e']
    
    bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
    ax3.set_title('Total Keywords Found by Category', fontsize=12)
    ax3.set_ylabel('Number of Keywords', fontsize=10)
    ax3.tick_params(axis='x', labelsize=9)
    ax3.tick_params(axis='y', labelsize=9)
    
    # Add value labels (smaller font)
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Processing Efficiency
    human_time = report['processing_efficiency']['estimated_human_time'] / 60  # minutes
    machine_time = report['processing_efficiency']['total_processing_time'] / 60  # minutes
    
    categories = ['Human (Estimated)', 'Machine (Actual)']
    times = [human_time, machine_time]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax4.bar(categories, times, color=colors, alpha=0.8)
    ax4.set_title('Processing Time Comparison', fontsize=12)
    ax4.set_ylabel('Time (minutes)', fontsize=10)
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', labelsize=9)
    ax4.tick_params(axis='y', labelsize=9)
    
    # Add value labels (smaller font)
    for bar, time_val in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f} min', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('cardio_resp_results/simple_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

def generate_summary_report(report, results, cardio_cases):
    """Generate summary report"""
    print("\n" + "="*80)
    print("📊 CARDIOVASCULAR AND RESPIRATORY CASE CLASSIFICATION RESULTS")
    print("="*80)
    
    # Basic statistics
    print(f"📋 Dataset Information:")
    print(f"   • Total Cases Processed: {report['dataset_info']['total_cases']:,}")
    print(f"   • Cardio-Respiratory Only: {report['dataset_info']['cardio_respiratory_only']:,} ({report['dataset_info']['cardio_resp_percentage']:.1f}%)")
    print(f"   • Mixed/Other Cases: {report['dataset_info']['mixed_or_other']:,}")
    
    print(f"\n⚡ Processing Efficiency:")
    print(f"   • Total Processing Time: {report['processing_efficiency']['total_processing_time']:.2f} seconds")
    print(f"   • Average Time per Case: {report['processing_efficiency']['avg_time_per_case']:.3f} seconds")
    print(f"   • Cases per Second: {report['processing_efficiency']['cases_per_second']:.0f}")
    print(f"   • Machine Efficiency: {report['processing_efficiency']['machine_efficiency_multiplier']:.0f}x faster than human")
    
    print(f"\n🤖 Machine Learning Model Performance:")
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
        print()
    
    # Top cardio-respiratory cases
    print(f"🔍 Top 5 Cardio-Respiratory Cases (by confidence):")
    sorted_cases = sorted(cardio_cases, key=lambda x: x['confidence'], reverse=True)[:5]
    
    for i, case in enumerate(sorted_cases, 1):
        print(f"   {i}. {case['case_id']} (Confidence: {case['confidence']:.3f})")
        print(f"      - Cardiovascular Keywords: {case['cardio_keywords']}")
        print(f"      - Respiratory Keywords: {case['resp_keywords']}")
        print(f"      - Other Medical Keywords: {case['other_keywords']}")
        print(f"      - Text Length: {case['text_length']:,} characters")
        print()
    
    # Keyword analysis
    all_keywords = []
    for case in results:
        all_keywords.extend(case['found_keywords'])
    
    if all_keywords:
        keyword_counts = pd.Series(all_keywords).value_counts()
        top_keywords = keyword_counts.head(10)
        
        print(f"🔑 Most Common Keywords Found:")
        for keyword, count in top_keywords.items():
            print(f"   • {keyword}: {count} occurrences")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete! Results saved to cardio_resp_results/")
    print("="*80)

def main():
    """Main analysis function"""
    print("🔍 Loading classification results...")
    try:
        report, results, cardio_cases = load_results()
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print("📊 Creating visualizations...")
    try:
        create_simple_visualizations(report, results)
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        print("Continuing with text report...")
    
    print("📝 Generating summary report...")
    generate_summary_report(report, results, cardio_cases)

if __name__ == "__main__":
    main()
