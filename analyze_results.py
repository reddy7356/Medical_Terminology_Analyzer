#!/usr/bin/env python3
"""
Analysis and Visualization of Cardio-Respiratory Classification Results
Creates confusion matrices, performance comparisons, and efficiency analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
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

def create_confusion_matrix_visualization(report):
    """Create confusion matrix visualization for all models"""
    models = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices - Cardio-Respiratory Classification', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(models):
        row = i // 2
        col = i % 2
        
        # Parse confusion matrix from string
        cm_str = report['ml_model_performance'][model_name]['confusion_matrix']
        # Clean up the string and parse
        cm_str = cm_str.replace('[', '').replace(']', '').replace('\n', ' ')
        cm_values = [int(x) for x in cm_str.split() if x.isdigit()]
        cm = np.array(cm_values).reshape(2, 2)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                   xticklabels=['Mixed/Other', 'Cardio-Respiratory Only'],
                   yticklabels=['Mixed/Other', 'Cardio-Respiratory Only'])
        
        axes[row, col].set_title(f'{model_name.replace("_", " ").title()}\n'
                                f'Accuracy: {report["ml_model_performance"][model_name]["accuracy"]:.3f}\n'
                                f'F1-Score: {report["ml_model_performance"][model_name]["f1_score"]:.3f}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('cardio_resp_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_comparison(report):
    """Create performance comparison chart"""
    models = list(report['ml_model_performance'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ML Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        values = [report['ml_model_performance'][model][metric] for model in models]
        model_names = [model.replace('_', ' ').title() for model in models]
        
        bars = axes[row, col].bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_ylabel('Score')
        axes[row, col].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cardio_resp_results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_analysis(report):
    """Create efficiency analysis chart"""
    # Calculate human vs machine efficiency
    total_cases = report['dataset_info']['total_cases']
    processing_time = report['processing_efficiency']['total_processing_time']
    estimated_human_time = report['processing_efficiency']['estimated_human_time']
    
    # Create efficiency comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Processing time comparison
    categories = ['Human (Estimated)', 'Machine (Actual)']
    times = [estimated_human_time / 60, processing_time / 60]  # Convert to minutes
    
    bars1 = ax1.bar(categories, times, color=['#ff7f0e', '#2ca02c'])
    ax1.set_title('Processing Time Comparison\n(Human vs Machine)')
    ax1.set_ylabel('Time (minutes)')
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f} min', ha='center', va='bottom')
    
    # Efficiency multiplier
    efficiency_multiplier = estimated_human_time / max(processing_time, 1)
    ax2.bar(['Efficiency Gain'], [efficiency_multiplier], color='#2ca02c')
    ax2.set_title('Machine Efficiency Multiplier')
    ax2.set_ylabel('Times Faster')
    ax2.set_yscale('log')
    ax2.text(0, efficiency_multiplier + 10, f'{efficiency_multiplier:.0f}x', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cardio_resp_results/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_case_distribution_analysis(results):
    """Create case distribution analysis"""
    df = pd.DataFrame(results)
    
    # Classification distribution
    classification_counts = df['classification'].value_counts()
    
    # Keyword analysis
    cardio_keywords = df['cardio_keywords'].sum()
    resp_keywords = df['resp_keywords'].sum()
    other_keywords = df['other_keywords'].sum()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Case Distribution and Keyword Analysis', fontsize=16, fontweight='bold')
    
    # Classification distribution pie chart
    ax1.pie(classification_counts.values, labels=classification_counts.index, autopct='%1.1f%%',
           colors=['#ff7f0e', '#2ca02c'])
    ax1.set_title('Case Classification Distribution')
    
    # Keyword distribution
    keyword_categories = ['Cardiovascular', 'Respiratory', 'Other Medical']
    keyword_counts = [cardio_keywords, resp_keywords, other_keywords]
    
    bars = ax2.bar(keyword_categories, keyword_counts, color=['#d62728', '#1f77b4', '#ff7f0e'])
    ax2.set_title('Total Keywords Found by Category')
    ax2.set_ylabel('Number of Keywords')
    
    # Add value labels
    for bar, count in zip(bars, keyword_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Text length distribution
    ax3.hist(df['text_length'], bins=50, alpha=0.7, color='#2ca02c')
    ax3.set_title('Distribution of Case Text Length')
    ax3.set_xlabel('Text Length (characters)')
    ax3.set_ylabel('Number of Cases')
    
    # Word count distribution
    ax4.hist(df['word_count'], bins=50, alpha=0.7, color='#1f77b4')
    ax4.set_title('Distribution of Case Word Count')
    ax4.set_xlabel('Word Count')
    ax4.set_ylabel('Number of Cases')
    
    plt.tight_layout()
    plt.savefig('cardio_resp_results/case_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(report, results, cardio_cases):
    """Generate detailed analysis report"""
    report_text = f"""
# CARDIOVASCULAR AND RESPIRATORY CASE CLASSIFICATION ANALYSIS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- **Total Cases Analyzed**: {report['dataset_info']['total_cases']:,}
- **Cardio-Respiratory Only Cases**: {report['dataset_info']['cardio_respiratory_only']:,} ({report['dataset_info']['cardio_resp_percentage']:.1f}%)
- **Mixed/Other Cases**: {report['dataset_info']['mixed_or_other']:,}
- **Processing Time**: {report['processing_efficiency']['total_processing_time']:.2f} seconds
- **Machine Efficiency**: {report['processing_efficiency']['machine_efficiency_multiplier']:.0f}x faster than human

## MACHINE LEARNING MODEL PERFORMANCE

### Best Performing Model: SVM
- **Accuracy**: {report['ml_model_performance']['svm']['accuracy']:.3f}
- **Precision**: {report['ml_model_performance']['svm']['precision']:.3f}
- **Recall**: {report['ml_model_performance']['svm']['recall']:.3f}
- **F1-Score**: {report['ml_model_performance']['svm']['f1_score']:.3f}
- **Training Time**: {report['ml_model_performance']['svm']['training_time']:.2f} seconds

### All Models Performance Comparison:
"""
    
    for model_name, metrics in report['ml_model_performance'].items():
        report_text += f"""
**{model_name.replace('_', ' ').title()}**:
- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1_score']:.3f}
- Training Time: {metrics['training_time']:.2f}s
"""
    
    report_text += f"""

## EFFICIENCY ANALYSIS

### Human vs Machine Performance:
- **Estimated Human Processing Time**: {report['processing_efficiency']['estimated_human_time']/60:.1f} minutes
- **Actual Machine Processing Time**: {report['processing_efficiency']['total_processing_time']:.2f} seconds
- **Efficiency Gain**: {report['processing_efficiency']['machine_efficiency_multiplier']:.0f}x faster
- **Time Saved**: {(report['processing_efficiency']['estimated_human_time'] - report['processing_efficiency']['total_processing_time'])/60:.1f} minutes

### Processing Speed:
- **Cases per Second**: {report['processing_efficiency']['cases_per_second']:.0f}
- **Average Time per Case**: {report['processing_efficiency']['avg_time_per_case']:.3f} seconds

## CARDIO-RESPIRATORY CASES FOUND

### Top 10 Cardio-Respiratory Cases by Confidence:
"""
    
    # Sort cardio cases by confidence
    sorted_cases = sorted(cardio_cases, key=lambda x: x['confidence'], reverse=True)[:10]
    
    for i, case in enumerate(sorted_cases, 1):
        report_text += f"""
{i}. **{case['case_id']}** (Confidence: {case['confidence']:.3f})
   - Cardiovascular Keywords: {case['cardio_keywords']}
   - Respiratory Keywords: {case['resp_keywords']}
   - Other Medical Keywords: {case['other_keywords']}
   - Text Length: {case['text_length']:,} characters
   - Word Count: {case['word_count']:,} words
"""
    
    report_text += f"""

## KEYWORD ANALYSIS

### Most Common Keywords Found:
"""
    
    # Analyze keyword frequency
    all_keywords = []
    for case in results:
        all_keywords.extend(case['found_keywords'])
    
    keyword_counts = pd.Series(all_keywords).value_counts()
    top_keywords = keyword_counts.head(20)
    
    for keyword, count in top_keywords.items():
        report_text += f"- {keyword}: {count} occurrences\n"
    
    report_text += f"""

## CONCLUSION

The machine learning classification system successfully identified {report['dataset_info']['cardio_respiratory_only']} cases 
({report['dataset_info']['cardio_resp_percentage']:.1f}%) as containing only cardiovascular and respiratory keywords 
out of {report['dataset_info']['total_cases']:,} total cases.

The SVM model achieved the best performance with {report['ml_model_performance']['svm']['accuracy']:.1f}% accuracy 
and {report['ml_model_performance']['svm']['f1_score']:.3f} F1-score, demonstrating high reliability in 
distinguishing cardio-respiratory cases from mixed medical cases.

The system processed all cases in {report['processing_efficiency']['total_processing_time']:.2f} seconds, 
which is {report['processing_efficiency']['machine_efficiency_multiplier']:.0f}x faster than estimated human 
processing time, showcasing significant efficiency gains for large-scale medical case analysis.
"""
    
    # Save report
    with open('cardio_resp_results/detailed_analysis_report.md', 'w') as f:
        f.write(report_text)
    
    print("📊 Detailed analysis report saved to: cardio_resp_results/detailed_analysis_report.md")
    return report_text

def main():
    """Main analysis function"""
    print("🔍 Loading classification results...")
    report, results, cardio_cases = load_results()
    
    print("📊 Creating confusion matrix visualization...")
    create_confusion_matrix_visualization(report)
    
    print("📈 Creating performance comparison charts...")
    create_performance_comparison(report)
    
    print("⚡ Creating efficiency analysis...")
    create_efficiency_analysis(report)
    
    print("📋 Creating case distribution analysis...")
    create_case_distribution_analysis(results)
    
    print("📝 Generating detailed report...")
    report_text = generate_detailed_report(report, results, cardio_cases)
    
    print("\n" + "="*80)
    print("📊 ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Found {report['dataset_info']['cardio_respiratory_only']} cardio-respiratory cases")
    print(f"✅ Best model accuracy: {report['ml_model_performance']['svm']['accuracy']:.3f}")
    print(f"✅ Machine efficiency: {report['processing_efficiency']['machine_efficiency_multiplier']:.0f}x faster")
    print(f"✅ All visualizations and reports saved to: cardio_resp_results/")
    print("="*80)

if __name__ == "__main__":
    main()
