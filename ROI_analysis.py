#!/usr/bin/env python3
"""
ROI and Cost Savings Analysis for Cardio-Respiratory AI Classifier
Generates comprehensive financial analysis for hospital presentation
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class ROI_Analyzer:
    """Comprehensive ROI and cost savings analyzer"""
    
    def __init__(self, 
                 cases_per_month=1000,
                 anesthesiologist_hourly_rate=200,
                 human_minutes_per_case=3,
                 machine_seconds_per_case=0.003,
                 baseline_cancellation_rate=0.02,
                 cancellation_cr_share=0.30,
                 cancellation_avoidance_fraction=0.40,
                 cost_per_cancellation=3000,
                 discount_rate=0.08):
        """
        Initialize ROI analyzer with parameters
        
        Args:
            cases_per_month: Number of cases processed monthly
            anesthesiologist_hourly_rate: Hourly compensation rate ($)
            human_minutes_per_case: Manual review time per case (minutes)
            machine_seconds_per_case: AI processing time per case (seconds)
            baseline_cancellation_rate: Baseline cancellation rate (0-1)
            cancellation_cr_share: % of cancellations due to cardio-respiratory issues
            cancellation_avoidance_fraction: % of CR cancellations preventable with early detection
            cost_per_cancellation: Direct cost per cancelled surgery ($)
            discount_rate: Annual discount rate for NPV (0-1)
        """
        self.cases_per_month = cases_per_month
        self.anesthesiologist_hourly_rate = anesthesiologist_hourly_rate
        self.human_minutes_per_case = human_minutes_per_case
        self.machine_seconds_per_case = machine_seconds_per_case
        self.baseline_cancellation_rate = baseline_cancellation_rate
        self.cancellation_cr_share = cancellation_cr_share
        self.cancellation_avoidance_fraction = cancellation_avoidance_fraction
        self.cost_per_cancellation = cost_per_cancellation
        self.discount_rate = discount_rate
    
    def calculate_time_savings(self) -> Dict:
        """Calculate time savings from automation"""
        # Human time
        human_hours_per_case = self.human_minutes_per_case / 60
        total_human_hours = self.cases_per_month * human_hours_per_case
        
        # Machine time
        machine_hours_per_case = self.machine_seconds_per_case / 3600
        total_machine_hours = self.cases_per_month * machine_hours_per_case
        
        # Savings
        time_saved_hours = total_human_hours - total_machine_hours
        efficiency_multiplier = total_human_hours / total_machine_hours if total_machine_hours > 0 else float('inf')
        
        return {
            'human_hours_per_month': total_human_hours,
            'machine_hours_per_month': total_machine_hours,
            'time_saved_hours_per_month': time_saved_hours,
            'efficiency_multiplier': efficiency_multiplier
        }
    
    def calculate_labor_savings(self) -> Dict:
        """Calculate labor cost savings"""
        time_savings = self.calculate_time_savings()
        
        monthly_labor_cost_before = time_savings['human_hours_per_month'] * self.anesthesiologist_hourly_rate
        monthly_labor_cost_after = time_savings['machine_hours_per_month'] * self.anesthesiologist_hourly_rate
        monthly_labor_savings = monthly_labor_cost_before - monthly_labor_cost_after
        annual_labor_savings = monthly_labor_savings * 12
        
        return {
            'monthly_labor_cost_before': monthly_labor_cost_before,
            'monthly_labor_cost_after': monthly_labor_cost_after,
            'monthly_labor_savings': monthly_labor_savings,
            'annual_labor_savings': annual_labor_savings
        }
    
    def calculate_cancellation_savings(self) -> Dict:
        """Calculate savings from reduced surgical cancellations"""
        # Baseline cancellations
        total_cancellations = self.cases_per_month * self.baseline_cancellation_rate
        cr_cancellations = total_cancellations * self.cancellation_cr_share
        
        # Avoidable cancellations with early detection
        avoidable_cancellations = cr_cancellations * self.cancellation_avoidance_fraction
        
        # Cost savings
        monthly_cancellation_savings = avoidable_cancellations * self.cost_per_cancellation
        annual_cancellation_savings = monthly_cancellation_savings * 12
        
        return {
            'baseline_monthly_cancellations': total_cancellations,
            'cr_monthly_cancellations': cr_cancellations,
            'avoidable_monthly_cancellations': avoidable_cancellations,
            'monthly_cancellation_savings': monthly_cancellation_savings,
            'annual_cancellation_savings': annual_cancellation_savings
        }
    
    def estimate_implementation_costs(self) -> Dict:
        """Estimate one-time and ongoing implementation costs"""
        # One-time costs
        software_development = 50000  # Custom integration, testing
        hardware_infrastructure = 10000  # Server, database
        training_hours = 40  # Staff training
        training_cost = training_hours * self.anesthesiologist_hourly_rate * 5  # 5 staff members
        total_one_time_costs = software_development + hardware_infrastructure + training_cost
        
        # Annual ongoing costs
        software_maintenance = 10000  # Updates, bug fixes
        hosting_database = 6000  # Cloud/server costs
        model_retraining = 8000  # Quarterly retraining
        support_monitoring = 12000  # Part-time support staff
        total_annual_ongoing_costs = software_maintenance + hosting_database + model_retraining + support_monitoring
        
        monthly_ongoing_costs = total_annual_ongoing_costs / 12
        
        return {
            'one_time_costs': {
                'software_development': software_development,
                'hardware_infrastructure': hardware_infrastructure,
                'training_cost': training_cost,
                'total': total_one_time_costs
            },
            'annual_ongoing_costs': {
                'software_maintenance': software_maintenance,
                'hosting_database': hosting_database,
                'model_retraining': model_retraining,
                'support_monitoring': support_monitoring,
                'total': total_annual_ongoing_costs
            },
            'monthly_ongoing_costs': monthly_ongoing_costs
        }
    
    def calculate_net_savings(self) -> Dict:
        """Calculate net savings after costs"""
        labor_savings = self.calculate_labor_savings()
        cancellation_savings = self.calculate_cancellation_savings()
        costs = self.estimate_implementation_costs()
        
        # Monthly
        monthly_gross_savings = (labor_savings['monthly_labor_savings'] + 
                                cancellation_savings['monthly_cancellation_savings'])
        monthly_net_savings = monthly_gross_savings - costs['monthly_ongoing_costs']
        
        # Annual
        annual_gross_savings = (labor_savings['annual_labor_savings'] + 
                               cancellation_savings['annual_cancellation_savings'])
        annual_net_savings = annual_gross_savings - costs['annual_ongoing_costs']['total']
        
        return {
            'monthly_gross_savings': monthly_gross_savings,
            'monthly_net_savings': monthly_net_savings,
            'annual_gross_savings': annual_gross_savings,
            'annual_net_savings': annual_net_savings,
            'payback_months': costs['one_time_costs']['total'] / monthly_net_savings if monthly_net_savings > 0 else float('inf')
        }
    
    def three_year_projection(self) -> Dict:
        """Generate 3-year financial projection with adoption curve"""
        costs = self.estimate_implementation_costs()
        net_savings = self.calculate_net_savings()
        
        # Adoption curve: pilot (25%) -> rollout (60%) -> full (95%)
        adoption_schedule = {
            'Month 1-2': 0.25,
            'Month 3-5': 0.60,
            'Month 6-36': 0.95
        }
        
        monthly_projections = []
        cumulative_savings = -costs['one_time_costs']['total']  # Start with initial investment
        
        for month in range(1, 37):  # 36 months = 3 years
            if month <= 2:
                adoption_rate = 0.25
            elif month <= 5:
                adoption_rate = 0.60
            else:
                adoption_rate = 0.95
            
            monthly_savings = net_savings['monthly_net_savings'] * adoption_rate
            cumulative_savings += monthly_savings
            
            # Discounted cash flow
            discount_factor = (1 + self.discount_rate) ** (month / 12)
            discounted_savings = monthly_savings / discount_factor
            
            monthly_projections.append({
                'month': month,
                'adoption_rate': adoption_rate,
                'monthly_savings': monthly_savings,
                'cumulative_savings': cumulative_savings,
                'discounted_savings': discounted_savings
            })
        
        # NPV calculation
        npv = -costs['one_time_costs']['total'] + sum(p['discounted_savings'] for p in monthly_projections)
        
        # IRR calculation (approximate)
        cash_flows = [-costs['one_time_costs']['total']] + [p['monthly_savings'] for p in monthly_projections]
        irr = self.calculate_irr(cash_flows)
        
        return {
            'monthly_projections': monthly_projections,
            'npv': npv,
            'irr': irr,
            'total_3_year_savings': cumulative_savings,
            'payback_month': next((p['month'] for p in monthly_projections if p['cumulative_savings'] > 0), None)
        }
    
    def calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using numpy"""
        try:
            return float(np.irr(cash_flows))
        except:
            # Approximate IRR if numpy.irr is not available
            return self.approximate_irr(cash_flows)
    
    def approximate_irr(self, cash_flows: List[float]) -> float:
        """Approximate IRR using binary search"""
        def npv_at_rate(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
        
        # Binary search for rate where NPV = 0
        low, high = -0.99, 10.0
        for _ in range(100):
            mid = (low + high) / 2
            npv_mid = npv_at_rate(mid)
            if abs(npv_mid) < 1:
                return mid
            if npv_mid > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2
    
    def sensitivity_analysis(self) -> Dict:
        """Perform sensitivity analysis on key parameters"""
        base_net_savings = self.calculate_net_savings()['annual_net_savings']
        
        sensitivities = {}
        
        # Vary hourly rate ±30%
        original_rate = self.anesthesiologist_hourly_rate
        for pct in [-0.3, -0.15, 0, 0.15, 0.3]:
            self.anesthesiologist_hourly_rate = original_rate * (1 + pct)
            net = self.calculate_net_savings()['annual_net_savings']
            sensitivities[f'rate_{pct:+.0%}'] = net
        self.anesthesiologist_hourly_rate = original_rate
        
        # Vary cancellation avoidance ±50%
        original_avoid = self.cancellation_avoidance_fraction
        avoid_sensitivities = {}
        for pct in [-0.5, -0.25, 0, 0.25, 0.5]:
            new_avoid = max(0, min(1, original_avoid * (1 + pct)))
            self.cancellation_avoidance_fraction = new_avoid
            net = self.calculate_net_savings()['annual_net_savings']
            avoid_sensitivities[f'avoid_{pct:+.0%}'] = net
        self.cancellation_avoidance_fraction = original_avoid
        
        return {
            'base_annual_net_savings': base_net_savings,
            'hourly_rate_sensitivity': sensitivities,
            'cancellation_avoidance_sensitivity': avoid_sensitivities
        }
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive summary for presentation"""
        time_savings = self.calculate_time_savings()
        labor_savings = self.calculate_labor_savings()
        cancellation_savings = self.calculate_cancellation_savings()
        costs = self.estimate_implementation_costs()
        net_savings = self.calculate_net_savings()
        projection = self.three_year_projection()
        sensitivity = self.sensitivity_analysis()
        
        return {
            'assumptions': {
                'cases_per_month': self.cases_per_month,
                'anesthesiologist_hourly_rate': self.anesthesiologist_hourly_rate,
                'human_minutes_per_case': self.human_minutes_per_case,
                'machine_seconds_per_case': self.machine_seconds_per_case,
                'baseline_cancellation_rate': self.baseline_cancellation_rate,
                'cancellation_cr_share': self.cancellation_cr_share,
                'cancellation_avoidance_fraction': self.cancellation_avoidance_fraction,
                'cost_per_cancellation': self.cost_per_cancellation,
                'discount_rate': self.discount_rate
            },
            'time_savings': time_savings,
            'labor_savings': labor_savings,
            'cancellation_savings': cancellation_savings,
            'implementation_costs': costs,
            'net_savings': net_savings,
            'three_year_projection': projection,
            'sensitivity_analysis': sensitivity
        }
    
    def plot_visualizations(self, summary: Dict, output_dir: str = "roi"):
        """Generate visualization plots for presentation"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        projection = summary['three_year_projection']
        
        # 1. Cumulative Savings Over 3 Years
        fig, ax = plt.subplots(figsize=(12, 6))
        months = [p['month'] for p in projection['monthly_projections']]
        cumulative = [p['cumulative_savings'] for p in projection['monthly_projections']]
        
        ax.plot(months, cumulative, linewidth=2, color='#2E86AB', label='Cumulative Savings')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        
        # Mark payback point
        payback_month = projection['payback_month']
        if payback_month:
            ax.axvline(x=payback_month, color='green', linestyle='--', alpha=0.5)
            ax.text(payback_month, max(cumulative)*0.8, 
                   f'Payback: Month {payback_month}', 
                   rotation=0, ha='left', fontsize=10, color='green')
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Cumulative Savings ($)', fontsize=12)
        ax.set_title('3-Year Cumulative Savings Projection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        plt.tight_layout()
        plt.savefig(output_path / "cumulative_savings.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Monthly Cash Flow
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_savings = [p['monthly_savings'] for p in projection['monthly_projections']]
        colors = ['#A23B72' if m <= 2 else '#F18F01' if m <= 5 else '#2E86AB' for m in months]
        
        ax.bar(months, monthly_savings, color=colors, alpha=0.7)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Monthly Net Savings ($)', fontsize=12)
        ax.set_title('Monthly Net Savings (Adoption Phased)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Legend for phases
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#A23B72', alpha=0.7, label='Pilot (25% adoption)'),
            Patch(facecolor='#F18F01', alpha=0.7, label='Rollout (60% adoption)'),
            Patch(facecolor='#2E86AB', alpha=0.7, label='Full (95% adoption)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path / "monthly_cash_flow.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Savings Breakdown
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Labor Savings', 'Cancellation Savings', 'Ongoing Costs']
        values = [
            summary['labor_savings']['annual_labor_savings'],
            summary['cancellation_savings']['annual_cancellation_savings'],
            -summary['implementation_costs']['annual_ongoing_costs']['total']
        ]
        colors_breakdown = ['#2E86AB', '#A23B72', '#F18F01']
        
        ax.barh(categories, values, color=colors_breakdown)
        ax.set_xlabel('Annual Amount ($)', fontsize=12)
        ax.set_title('Annual Savings Breakdown (Year 2+)', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 2000, i, f'${abs(v)/1000:.1f}K', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / "savings_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Sensitivity Tornado Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rate_sens = summary['sensitivity_analysis']['hourly_rate_sensitivity']
        base_value = summary['sensitivity_analysis']['base_annual_net_savings']
        
        scenarios = ['-30%', '-15%', 'Base', '+15%', '+30%']
        rate_values = [rate_sens[k] for k in ['rate_-30%', 'rate_-15%']] + [base_value] + [rate_sens[k] for k in ['rate_+15%', 'rate_+30%']]
        
        y_pos = np.arange(len(scenarios))
        colors_sens = ['#C84B31' if v < base_value else '#2E86AB' for v in rate_values]
        
        ax.barh(y_pos, rate_values, color=colors_sens, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenarios)
        ax.set_xlabel('Annual Net Savings ($)', fontsize=12)
        ax.set_ylabel('Hourly Rate Variation', fontsize=12)
        ax.set_title('Sensitivity Analysis: Anesthesiologist Hourly Rate', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.axvline(x=base_value, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "sensitivity_tornado.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description='ROI and Cost Savings Analysis')
    parser.add_argument('--cases', type=int, default=1000, help='Cases per month')
    parser.add_argument('--rate', type=float, default=200, help='Anesthesiologist hourly rate ($)')
    parser.add_argument('--cancel-rate', type=float, default=0.02, help='Baseline cancellation rate')
    parser.add_argument('--cr-share', type=float, default=0.30, help='CR share of cancellations')
    parser.add_argument('--avoid', type=float, default=0.40, help='Cancellation avoidance fraction')
    parser.add_argument('--cancel-cost', type=float, default=3000, help='Cost per cancellation ($)')
    parser.add_argument('--output-dir', type=str, default='roi', help='Output directory')
    
    args = parser.parse_args()
    
    print("🏥 ROI and Cost Savings Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ROI_Analyzer(
        cases_per_month=args.cases,
        anesthesiologist_hourly_rate=args.rate,
        baseline_cancellation_rate=args.cancel_rate,
        cancellation_cr_share=args.cr_share,
        cancellation_avoidance_fraction=args.avoid,
        cost_per_cancellation=args.cancel_cost
    )
    
    # Generate summary
    summary = analyzer.generate_summary()
    
    # Save summary
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "ROI_summary.json", 'w') as f:
        # Convert for JSON serialization
        serializable_summary = json.loads(json.dumps(summary, default=str))
        json.dump(serializable_summary, f, indent=2)
    
    # Generate visualizations
    analyzer.plot_visualizations(summary, args.output_dir)
    
    # Print key metrics
    print("\n📊 KEY METRICS")
    print("-" * 80)
    print(f"Monthly Cases: {args.cases:,}")
    print(f"Time Saved: {summary['time_savings']['time_saved_hours_per_month']:.1f} hours/month")
    print(f"Efficiency Gain: {summary['time_savings']['efficiency_multiplier']:,.0f}x faster")
    print()
    print(f"Monthly Labor Savings: ${summary['labor_savings']['monthly_labor_savings']:,.0f}")
    print(f"Monthly Cancellation Savings: ${summary['cancellation_savings']['monthly_cancellation_savings']:,.0f}")
    print(f"Monthly Gross Savings: ${summary['net_savings']['monthly_gross_savings']:,.0f}")
    print(f"Monthly Net Savings: ${summary['net_savings']['monthly_net_savings']:,.0f}")
    print()
    print(f"Implementation Cost: ${summary['implementation_costs']['one_time_costs']['total']:,.0f}")
    print(f"Payback Period: {summary['net_savings']['payback_months']:.1f} months")
    print()
    print(f"3-Year Total Savings: ${summary['three_year_projection']['total_3_year_savings']:,.0f}")
    print(f"NPV (8% discount): ${summary['three_year_projection']['npv']:,.0f}")
    print(f"IRR: {summary['three_year_projection']['irr']*100:.1f}%")
    print("=" * 80)
    print(f"\n💾 Full analysis saved to {output_path}/ROI_summary.json")
    print(f"📊 Visualizations saved to {output_path}/")


if __name__ == "__main__":
    main()
