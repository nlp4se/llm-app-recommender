import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def rbo(list1, list2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two lists.
    p is the persistence parameter, giving weight to top-ranked items.
    """
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    
    if not list1 or not list2:
        return 0

    s_len, t_len = len(list1), len(list2)
    max_depth = max(s_len, t_len)
    
    # Calculate agreement at each depth
    agreement = 0.0
    
    for d in range(1, max_depth + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        agreement_d = len(set1.intersection(set2)) / d
        agreement += agreement_d * (p ** (d-1))

    return (1 - p) * agreement

class RBOAnalyzer:
    def __init__(self, blind_recommendations_path, guided_recommendations_path):
        """
        Initialize the RBO analyzer with two CSV files:
        - blind_recommendations: recommendations without specific ranking criteria
        - guided_recommendations: recommendations with specific ranking criteria
        """
        self.blind_df = pd.read_csv(blind_recommendations_path)
        self.guided_df = pd.read_csv(guided_recommendations_path)
        
        # Create output directory
        self.output_dir = Path('data/output/evaluation/correlation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def preprocess_data(self):
        """Preprocess and clean the data"""
        print("Preprocessing data...")
        
        # Clean column names and handle missing values
        self.blind_df = self.blind_df.fillna('')
        self.guided_df = self.guided_df.fillna('')
        
        # Get unique models, features, and criteria - preserve order from input files
        self.models = self.blind_df['model'].unique()
        self.features = self.blind_df['feature'].unique()
        self.criteria = self.guided_df['ranking_criteria'].unique()
        
        print(f"Found {len(self.models)} models: {self.models}")
        print(f"Found {len(self.features)} features: {self.features}")
        print(f"Found {len(self.criteria)} ranking criteria: {self.criteria}")
        
    def model_specific_analysis(self):
        """
        Model-specific analysis: RBO between blind recommendations and each ranking criteria
        """
        print("\nPerforming model-specific analysis...")
        
        results = []
        
        for model in self.models:
            print(f"Processing model: {model}")
            
            # Get blind recommendations for this model
            blind_model = self.blind_df[self.blind_df['model'] == model]
            
            for feature in self.features:
                # Get blind recommendations for this feature
                blind_feature = blind_model[blind_model['feature'] == feature]
                
                if blind_feature.empty:
                    continue
                
                # Get guided recommendations for this model and feature
                guided_feature = self.guided_df[
                    (self.guided_df['model'] == model) & 
                    (self.guided_df['feature'] == feature)
                ]
                
                if guided_feature.empty:
                    continue
                
                # Calculate RBO for each ranking criteria
                for criteria in self.criteria:
                    guided_criteria = guided_feature[guided_feature['ranking_criteria'] == criteria]
                    
                    if guided_criteria.empty:
                        continue
                    
                    # Get ranking lists (columns 1-20)
                    rank_columns = [str(i) for i in range(1, 21)]
                    
                    # Calculate average RBO across runs
                    rbo_scores = []
                    for _, blind_row in blind_feature.iterrows():
                        blind_list = [blind_row[col] for col in rank_columns if blind_row[col]]
                        
                        for _, guided_row in guided_criteria.iterrows():
                            guided_list = [guided_row[col] for col in rank_columns if guided_row[col]]
                            
                            if blind_list and guided_list:
                                rbo_score = rbo(blind_list, guided_list, p=0.95)
                                rbo_scores.append(rbo_score)
                    
                    if rbo_scores:
                        avg_rbo = np.mean(rbo_scores)
                        results.append({
                            'model': model,
                            'feature': feature,
                            'criteria': criteria,
                            'rbo': avg_rbo,
                            'num_comparisons': len(rbo_scores)
                        })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'model_specific_rbo.csv', index=False)
        
        # Create visualization
        self.plot_model_specific_rbo(results_df)
        
        return results_df
    
    def plot_model_specific_rbo(self, results_df):
        """Create heatmap for model-specific RBO"""
        if results_df.empty:
            print("No data for model-specific RBO plot")
            return
            
        # Pivot data for heatmap - preserve model order from input files
        pivot_data = results_df.pivot_table(
            values='rbo', 
            index='criteria', 
            columns='model', 
            aggfunc='mean'
        )
        
        # Reorder columns to match the original model order
        pivot_data = pivot_data.reindex(columns=self.models)
        
        # Sort criteria alphabetically
        pivot_data_sorted = pivot_data.sort_index()
        
        # Add empty row for visual separation
        empty_row = pd.DataFrame([[np.nan] * len(pivot_data_sorted.columns)], 
                               index=[''], 
                               columns=pivot_data_sorted.columns)
        
        # Add average row at the bottom
        overall_avg = pivot_data_sorted.mean(axis=0)
        avg_row = pd.DataFrame([overall_avg], index=['AVERAGE RBO'])
        
        # Concatenate: criteria + empty row + average
        pivot_data_with_avg = pd.concat([pivot_data_sorted, empty_row, avg_row])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with YlGnBu color palette (same as consistency analysis)
        sns.heatmap(
            pivot_data_with_avg, 
            annot=True, 
            cmap='YlGnBu', 
            vmin=0, vmax=1,
            fmt='.2f',
            cbar_kws={'label': 'RBO Score'},
            ax=ax
        )
        
        ax.set_title('Model-Specific RBO: Blind vs Guided Recommendations', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_specific_rbo_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate summary table with averages
        summary_data = pd.DataFrame({
            'Average RBO': overall_avg
        })
        summary_data.to_csv(self.output_dir / 'model_averages_summary.csv')
        
        print(f"Model-specific RBO heatmap saved to: {self.output_dir / 'model_specific_rbo_heatmap.png'}")
        print(f"Model averages summary saved to: {self.output_dir / 'model_averages_summary.csv'}")
    
    def criteria_specific_analysis(self):
        """
        Criteria-specific analysis: compare how different models perform for each criteria
        """
        print("\nPerforming criteria-specific analysis...")
        
        results = []
        
        for criteria in self.criteria:
            print(f"Processing criteria: {criteria}")
            
            # Get guided recommendations for this criteria
            guided_criteria = self.guided_df[self.guided_df['ranking_criteria'] == criteria]
            
            for feature in self.features:
                guided_feature = guided_criteria[guided_criteria['feature'] == feature]
                
                if guided_feature.empty:
                    continue
                
                # Compare each model's blind recommendations with this criteria
                for model in self.models:
                    blind_model_feature = self.blind_df[
                        (self.blind_df['model'] == model) & 
                        (self.blind_df['feature'] == feature)
                    ]
                    
                    if blind_model_feature.empty:
                        continue
                    
                    # Calculate RBO
                    rank_columns = [str(i) for i in range(1, 21)]
                    rbo_scores = []
                    
                    for _, blind_row in blind_model_feature.iterrows():
                        blind_list = [blind_row[col] for col in rank_columns if blind_row[col]]
                        
                        for _, guided_row in guided_feature.iterrows():
                            guided_list = [guided_row[col] for col in rank_columns if guided_row[col]]
                            
                            if blind_list and guided_list:
                                rbo_score = rbo(blind_list, guided_list, p=0.95)
                                rbo_scores.append(rbo_score)
                    
                    if rbo_scores:
                        avg_rbo = np.mean(rbo_scores)
                        results.append({
                            'criteria': criteria,
                            'feature': feature,
                            'model': model,
                            'rbo': avg_rbo,
                            'num_comparisons': len(rbo_scores)
                        })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'criteria_specific_rbo.csv', index=False)
        
        # Create visualization
        self.plot_criteria_specific_rbo(results_df)
        
        return results_df
    
    def plot_criteria_specific_rbo(self, results_df):
        """Create box plots for criteria-specific RBO"""
        if results_df.empty:
            print("No data for criteria-specific RBO plot")
            return
            
        # Create figure with subplots - changed to 6 columns
        n_criteria = len(results_df['criteria'].unique())
        n_cols = 6
        n_rows = (n_criteria + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, criteria in enumerate(sorted(results_df['criteria'].unique())):
            if i >= len(axes):
                break
                
            ax = axes[i]
            criteria_data = results_df[results_df['criteria'] == criteria]
            
            # Create box plot
            sns.boxplot(data=criteria_data, x='model', y='rbo', ax=ax)
            ax.set_title(f'{criteria}', fontsize=12)
            ax.set_xlabel('Model')
            ax.set_ylabel('RBO')
            ax.tick_params(axis='x', rotation=45)
            
            # Set y-axis limits to 0 and 1 for consistent scale
            ax.set_ylim(0, 1)
            
            # Add mean points with 2 decimal places
            means = criteria_data.groupby('model')['rbo'].mean()
            for j, model in enumerate(means.index):
                ax.scatter(j, means[model], color='red', s=50, zorder=5, marker='o')
                # Add mean value annotation with 2 decimal places
                ax.annotate(f'{means[model]:.2f}', 
                           (j, means[model]), 
                           xytext=(0, 10), 
                           textcoords='offset points', 
                           ha='center', 
                           fontsize=8)
        
        # Hide empty subplots
        for i in range(len(results_df['criteria'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Criteria-Specific RBO: Model Performance by Ranking Criteria', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'criteria_specific_rbo_boxplot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Criteria-specific RBO boxplot saved to: {self.output_dir / 'criteria_specific_rbo_boxplot.png'}")
    
    def inter_criteria_analysis(self):
        """
        Inter-criteria analysis: find RBO between different ranking criteria
        """
        print("\nPerforming inter-criteria analysis...")
        
        # Store results for each model and overall average
        model_results = {}
        all_rbo_scores = {}
        
        for model in self.models:
            print(f"Processing inter-criteria analysis for model: {model}")
            
            # Create a matrix to store criteria RBO for this model
            criteria_matrix = {}
            
            for criteria1 in self.criteria:
                criteria_matrix[criteria1] = {}
                for criteria2 in self.criteria:
                    if criteria1 == criteria2:
                        criteria_matrix[criteria1][criteria2] = 1.0
                    else:
                        # Calculate RBO between criteria1 and criteria2
                        rbo_scores = []
                        
                        for feature in self.features:
                            # Get recommendations for both criteria
                            guided1 = self.guided_df[
                                (self.guided_df['model'] == model) & 
                                (self.guided_df['feature'] == feature) & 
                                (self.guided_df['ranking_criteria'] == criteria1)
                            ]
                            
                            guided2 = self.guided_df[
                                (self.guided_df['model'] == model) & 
                                (self.guided_df['feature'] == feature) & 
                                (self.guided_df['ranking_criteria'] == criteria2)
                            ]
                            
                            if guided1.empty or guided2.empty:
                                continue
                            
                            # Calculate RBO between rankings
                            rank_columns = [str(i) for i in range(1, 21)]
                            
                            for _, row1 in guided1.iterrows():
                                list1 = [row1[col] for col in rank_columns if row1[col]]
                                
                                for _, row2 in guided2.iterrows():
                                    list2 = [row2[col] for col in rank_columns if row2[col]]
                                    
                                    if list1 and list2:
                                        rbo_score = rbo(list1, list2, p=0.95)
                                        rbo_scores.append(rbo_score)
                        
                        if rbo_scores:
                            avg_rbo = np.mean(rbo_scores)
                            criteria_matrix[criteria1][criteria2] = avg_rbo
                        else:
                            criteria_matrix[criteria1][criteria2] = 0.0
            
            # Store results for this model
            model_results[model] = pd.DataFrame(criteria_matrix)
            
            # Accumulate RBO scores for overall average
            for criteria1 in self.criteria:
                if criteria1 not in all_rbo_scores:
                    all_rbo_scores[criteria1] = {}
                for criteria2 in self.criteria:
                    if criteria2 not in all_rbo_scores[criteria1]:
                        all_rbo_scores[criteria1][criteria2] = []
                    all_rbo_scores[criteria1][criteria2].append(criteria_matrix[criteria1][criteria2])
        
        # Calculate overall average
        overall_matrix = {}
        for criteria1 in self.criteria:
            overall_matrix[criteria1] = {}
            for criteria2 in self.criteria:
                overall_matrix[criteria1][criteria2] = np.mean(all_rbo_scores[criteria1][criteria2])
        
        overall_df = pd.DataFrame(overall_matrix)
        
        # Save results
        overall_df.to_csv(self.output_dir / 'inter_criteria_rbo_overall.csv')
        
        # Create visualizations
        self.plot_inter_criteria_rbo_by_model(model_results)
        self.plot_inter_criteria_rbo_overall(overall_df)
        
        return overall_df, model_results
    
    def plot_inter_criteria_rbo_by_model(self, model_results):
        """Create heatmaps for inter-criteria RBO for each model and save as separate files"""
        if not model_results:
            print("No data for model-specific inter-criteria RBO plots")
            return
        
        # Create individual plots for each model
        for model, criteria_df in model_results.items():
            # Create figure for this model
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap for this model with YlGnBu color palette
            sns.heatmap(
                criteria_df, 
                annot=True, 
                cmap='YlGnBu', 
                vmin=0, vmax=1,
                fmt='.2f',
                cbar_kws={'label': 'RBO Score'},
                ax=ax,
                square=True
            )
            
            ax.set_title(f'Inter-Criteria RBO: {model}', fontsize=16, pad=20)
            ax.set_xlabel('Ranking Criteria', fontsize=12)
            ax.set_ylabel('Ranking Criteria', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save individual file for this model
            clean_model_name = model.replace(' ', '_').replace('-', '_').replace('+', 'plus')
            filename = f'inter_criteria_rbo_{clean_model_name}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Inter-criteria RBO heatmap for {model} saved to: {self.output_dir / filename}")
    
    def plot_inter_criteria_rbo_overall(self, criteria_df):
        """Create heatmap for overall inter-criteria RBO"""
        if criteria_df.empty:
            print("No data for overall inter-criteria RBO plot")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap with YlGnBu color palette
        sns.heatmap(
            criteria_df, 
            annot=True, 
            cmap='YlGnBu', 
            vmin=0, vmax=1,
            fmt='.2f',
            cbar_kws={'label': 'RBO Score'},
            ax=ax,
            square=True
        )
        
        ax.set_title('Overall Inter-Criteria RBO: Average Across All Models', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Ranking Criteria', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inter_criteria_rbo_overall.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overall inter-criteria RBO heatmap saved to: {self.output_dir / 'inter_criteria_rbo_overall.png'}")
    
    def generate_summary_report(self, model_results, criteria_results, inter_criteria_results):
        """Generate a comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("RBO ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model-specific summary
        report.append("1. MODEL-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        if not model_results.empty:
            # Group by model and calculate averages
            model_groups = model_results.groupby('model')
            report.append("Average RBO by model (blind vs guided recommendations):")
            for model, group in model_groups:
                avg_rbo = group['rbo'].mean()
                report.append(f"  {model}: {avg_rbo:.3f} (n={len(group)})")
            
            # Criteria-specific averages
            criteria_groups = model_results.groupby('criteria')
            report.append("\nAverage RBO by ranking criteria:")
            for criteria, group in list(criteria_groups)[:10]:
                avg_rbo = group['rbo'].mean()
                report.append(f"  {criteria}: {avg_rbo:.3f} (n={len(group)})")
        report.append("")
        
        # Criteria-specific summary
        report.append("2. CRITERIA-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        if not criteria_results.empty:
            criteria_model_groups = criteria_results.groupby('criteria')
            report.append("Best performing criteria (highest average RBO):")
            for criteria, group in list(criteria_model_groups)[:5]:
                avg_rbo = group['rbo'].mean()
                report.append(f"  {criteria}: {avg_rbo:.3f} (n={len(group)})")
            
            model_criteria_groups = criteria_results.groupby('model')
            report.append("\nBest performing models (highest average RBO):")
            for model, group in list(model_criteria_groups)[:5]:
                avg_rbo = group['rbo'].mean()
                report.append(f"  {model}: {avg_rbo:.3f} (n={len(group)})")
        report.append("")
        
        # Inter-criteria summary
        report.append("3. INTER-CRITERIA ANALYSIS")
        report.append("-" * 40)
        if not inter_criteria_results.empty:
            # Find most similar criteria pairs
            similar_pairs = []
            for i, criteria1 in enumerate(inter_criteria_results.index):
                for j, criteria2 in enumerate(inter_criteria_results.columns):
                    if i < j:  # Avoid duplicates and self-correlations
                        rbo_score = inter_criteria_results.loc[criteria1, criteria2]
                        similar_pairs.append((criteria1, criteria2, rbo_score))
            
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            report.append("Most similar ranking criteria pairs:")
            for criteria1, criteria2, rbo_score in similar_pairs[:10]:
                report.append(f"  {criteria1} ↔ {criteria2}: {rbo_score:.3f}")
        report.append("")
        
        # Key insights
        report.append("4. KEY INSIGHTS")
        report.append("-" * 40)
        if not model_results.empty:
            overall_avg = model_results['rbo'].mean()
            report.append(f"• Overall average RBO: {overall_avg:.3f}")
            
            if overall_avg > 0.7:
                report.append("• High RBO suggests LLMs are consistent in their recommendations")
            elif overall_avg > 0.4:
                report.append("• Moderate RBO suggests some consistency with room for improvement")
            else:
                report.append("• Low RBO suggests significant differences between blind and guided recommendations")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'rbo_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nDetailed report saved to: {self.output_dir / 'rbo_analysis_report.txt'}")
    
    def run_analysis(self):
        """Run the complete RBO analysis"""
        print("Starting RBO analysis...")
        print(f"Blind recommendations: {len(self.blind_df)} rows")
        print(f"Guided recommendations: {len(self.guided_df)} rows")
        
        # Preprocess data
        self.preprocess_data()
        
        # Run all analyses
        model_results = self.model_specific_analysis()
        criteria_results = self.criteria_specific_analysis()
        inter_criteria_results, model_inter_results = self.inter_criteria_analysis()
        
        # Generate summary report
        self.generate_summary_report(model_results, criteria_results, inter_criteria_results)
        
        print("\nRBO analysis completed!")
        print(f"All results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze RBO between blind and guided recommendations')
    parser.add_argument('--blind-recommendations', type=str, 
                       default='data/output/evaluation/app_rankings.csv',
                       help='Path to blind recommendations CSV file')
    parser.add_argument('--guided-recommendations', type=str,
                       default='data/output/evaluation/correlation/app_rankings.csv',
                       help='Path to guided recommendations CSV file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.blind_recommendations).exists():
        print(f"Error: Blind recommendations file not found: {args.blind_recommendations}")
        return
    
    if not Path(args.guided_recommendations).exists():
        print(f"Error: Guided recommendations file not found: {args.guided_recommendations}")
        return
    
    # Run analysis
    analyzer = RBOAnalyzer(args.blind_recommendations, args.guided_recommendations)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
