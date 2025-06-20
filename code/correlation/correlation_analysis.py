import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    def __init__(self, blind_recommendations_path, guided_recommendations_path):
        """
        Initialize the correlation analyzer with two CSV files:
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
        
    def calculate_ranking_correlation(self, list1, list2, method='spearman'):
        """
        Calculate correlation between two ranking lists
        """
        if not list1 or not list2:
            return 0.0
            
        # Create a set of all unique apps
        all_apps = set(list1) | set(list2)
        
        # Create ranking vectors (1-based indexing)
        rank1 = {app: i+1 for i, app in enumerate(list1)}
        rank2 = {app: i+1 for i, app in enumerate(list2)}
        
        # Fill missing apps with a high rank (worse than any existing)
        max_rank = max(len(list1), len(list2)) + 1
        for app in all_apps:
            if app not in rank1:
                rank1[app] = max_rank
            if app not in rank2:
                rank2[app] = max_rank
        
        # Create vectors for correlation calculation
        apps_list = list(all_apps)
        vector1 = [rank1[app] for app in apps_list]
        vector2 = [rank2[app] for app in apps_list]
        
        if method == 'spearman':
            correlation, _ = spearmanr(vector1, vector2)
        elif method == 'pearson':
            correlation, _ = pearsonr(vector1, vector2)
        elif method == 'cosine':
            # Convert to similarity (inverse of rank)
            sim1 = [1/rank for rank in vector1]
            sim2 = [1/rank for rank in vector2]
            correlation = cosine_similarity([sim1], [sim2])[0][0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        return correlation if not np.isnan(correlation) else 0.0
    
    def model_specific_analysis(self):
        """
        Model-specific analysis: correlation between blind recommendations and each ranking criteria
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
                
                # Calculate correlation for each ranking criteria
                for criteria in self.criteria:
                    guided_criteria = guided_feature[guided_feature['ranking_criteria'] == criteria]
                    
                    if guided_criteria.empty:
                        continue
                    
                    # Get ranking lists (columns 1-20)
                    rank_columns = [str(i) for i in range(1, 21)]
                    
                    # Calculate average correlation across runs
                    correlations = []
                    for _, blind_row in blind_feature.iterrows():
                        blind_list = [blind_row[col] for col in rank_columns if blind_row[col]]
                        
                        for _, guided_row in guided_criteria.iterrows():
                            guided_list = [guided_row[col] for col in rank_columns if guided_row[col]]
                            
                            if blind_list and guided_list:
                                corr = self.calculate_ranking_correlation(blind_list, guided_list)
                                correlations.append(corr)
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        results.append({
                            'model': model,
                            'feature': feature,
                            'criteria': criteria,
                            'correlation': avg_correlation,
                            'num_comparisons': len(correlations)
                        })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'model_specific_correlations.csv', index=False)
        
        # Create visualization
        self.plot_model_specific_correlations(results_df)
        
        return results_df
    
    def plot_model_specific_correlations(self, results_df):
        """Create heatmap for model-specific correlations"""
        if results_df.empty:
            print("No data for model-specific correlation plot")
            return
            
        # Pivot data for heatmap - preserve model order from input files
        pivot_data = results_df.pivot_table(
            values='correlation', 
            index='criteria', 
            columns='model', 
            aggfunc='mean'
        )
        
        # Reorder columns to match the original model order
        pivot_data = pivot_data.reindex(columns=self.models)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with fixed scale from -1 to 1
        sns.heatmap(
            pivot_data, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            vmin=-1, vmax=1,
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title('Model-Specific Correlations: Blind vs Guided Recommendations', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_specific_correlations_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model-specific correlation heatmap saved to: {self.output_dir / 'model_specific_correlations_heatmap.png'}")
    
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
                    
                    # Calculate correlation
                    rank_columns = [str(i) for i in range(1, 21)]
                    correlations = []
                    
                    for _, blind_row in blind_model_feature.iterrows():
                        blind_list = [blind_row[col] for col in rank_columns if blind_row[col]]
                        
                        for _, guided_row in guided_feature.iterrows():
                            guided_list = [guided_row[col] for col in rank_columns if guided_row[col]]
                            
                            if blind_list and guided_list:
                                corr = self.calculate_ranking_correlation(blind_list, guided_list)
                                correlations.append(corr)
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        results.append({
                            'criteria': criteria,
                            'feature': feature,
                            'model': model,
                            'correlation': avg_correlation,
                            'num_comparisons': len(correlations)
                        })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'criteria_specific_correlations.csv', index=False)
        
        # Create visualization
        self.plot_criteria_specific_correlations(results_df)
        
        return results_df
    
    def plot_criteria_specific_correlations(self, results_df):
        """Create box plots for criteria-specific correlations"""
        if results_df.empty:
            print("No data for criteria-specific correlation plot")
            return
            
        # Create figure with subplots - fixed to 4 columns
        n_criteria = len(results_df['criteria'].unique())
        n_cols = 4
        n_rows = (n_criteria + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
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
            sns.boxplot(data=criteria_data, x='model', y='correlation', ax=ax)
            ax.set_title(f'{criteria}', fontsize=12)
            ax.set_xlabel('Model')
            ax.set_ylabel('Correlation')
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean points
            means = criteria_data.groupby('model')['correlation'].mean()
            for j, model in enumerate(means.index):
                ax.scatter(j, means[model], color='red', s=50, zorder=5, marker='o')
        
        # Hide empty subplots
        for i in range(len(results_df['criteria'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Criteria-Specific Correlations: Model Performance by Ranking Criteria', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'criteria_specific_correlations_boxplot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Criteria-specific correlation boxplot saved to: {self.output_dir / 'criteria_specific_correlations_boxplot.png'}")
    
    def inter_criteria_analysis(self):
        """
        Inter-criteria analysis: find correlations between different ranking criteria
        """
        print("\nPerforming inter-criteria analysis...")
        
        # Store results for each model and overall average
        model_results = {}
        all_correlations = {}
        
        for model in self.models:
            print(f"Processing inter-criteria analysis for model: {model}")
            
            # Create a matrix to store criteria correlations for this model
            criteria_matrix = {}
            
            for criteria1 in self.criteria:
                criteria_matrix[criteria1] = {}
                for criteria2 in self.criteria:
                    if criteria1 == criteria2:
                        criteria_matrix[criteria1][criteria2] = 1.0
                    else:
                        # Calculate correlation between criteria1 and criteria2
                        correlations = []
                        
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
                            
                            # Calculate correlation between rankings
                            rank_columns = [str(i) for i in range(1, 21)]
                            
                            for _, row1 in guided1.iterrows():
                                list1 = [row1[col] for col in rank_columns if row1[col]]
                                
                                for _, row2 in guided2.iterrows():
                                    list2 = [row2[col] for col in rank_columns if row2[col]]
                                    
                                    if list1 and list2:
                                        corr = self.calculate_ranking_correlation(list1, list2)
                                        correlations.append(corr)
                        
                        if correlations:
                            avg_correlation = np.mean(correlations)
                            criteria_matrix[criteria1][criteria2] = avg_correlation
                        else:
                            criteria_matrix[criteria1][criteria2] = 0.0
            
            # Store results for this model
            model_results[model] = pd.DataFrame(criteria_matrix)
            
            # Accumulate correlations for overall average
            for criteria1 in self.criteria:
                if criteria1 not in all_correlations:
                    all_correlations[criteria1] = {}
                for criteria2 in self.criteria:
                    if criteria2 not in all_correlations[criteria1]:
                        all_correlations[criteria1][criteria2] = []
                    all_correlations[criteria1][criteria2].append(criteria_matrix[criteria1][criteria2])
        
        # Calculate overall average
        overall_matrix = {}
        for criteria1 in self.criteria:
            overall_matrix[criteria1] = {}
            for criteria2 in self.criteria:
                overall_matrix[criteria1][criteria2] = np.mean(all_correlations[criteria1][criteria2])
        
        overall_df = pd.DataFrame(overall_matrix)
        
        # Save results
        overall_df.to_csv(self.output_dir / 'inter_criteria_correlations_overall.csv')
        
        # Create visualizations
        self.plot_inter_criteria_correlations_by_model(model_results)
        self.plot_inter_criteria_correlations_overall(overall_df)
        
        return overall_df, model_results
    
    def plot_inter_criteria_correlations_by_model(self, model_results):
        """Create heatmaps for inter-criteria correlations for each model"""
        if not model_results:
            print("No data for model-specific inter-criteria correlation plots")
            return
        
        # Create subplots for all models
        n_models = len(model_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (model, criteria_df) in enumerate(model_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create heatmap for this model
            sns.heatmap(
                criteria_df, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                vmin=-1, vmax=1,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax,
                square=True
            )
            
            ax.set_title(f'Inter-Criteria Correlations: {model}', fontsize=14)
            ax.set_xlabel('Ranking Criteria', fontsize=10)
            ax.set_ylabel('Ranking Criteria', fontsize=10)
            
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Hide empty subplots
        for i in range(len(model_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Inter-Criteria Correlations by Model', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inter_criteria_correlations_by_model.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model-specific inter-criteria correlation heatmaps saved to: {self.output_dir / 'inter_criteria_correlations_by_model.png'}")
    
    def plot_inter_criteria_correlations_overall(self, criteria_df):
        """Create heatmap for overall inter-criteria correlations"""
        if criteria_df.empty:
            print("No data for overall inter-criteria correlation plot")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap with fixed scale from -1 to 1
        sns.heatmap(
            criteria_df, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            vmin=-1, vmax=1,
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax,
            square=True
        )
        
        ax.set_title('Overall Inter-Criteria Correlations: Average Across All Models', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Ranking Criteria', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inter_criteria_correlations_overall.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overall inter-criteria correlation heatmap saved to: {self.output_dir / 'inter_criteria_correlations_overall.png'}")
    
    def calculate_confidence_intervals(self, correlations_list, confidence_level=0.95):
        """Calculate confidence intervals for correlation values"""
        if not correlations_list or len(correlations_list) < 2:
            return None, None, None
        
        # Calculate standard error of the mean
        mean_corr = np.mean(correlations_list)
        std_err = np.std(correlations_list, ddof=1) / np.sqrt(len(correlations_list))
        
        # Calculate confidence interval using t-distribution
        from scipy.stats import t
        t_value = t.ppf((1 + confidence_level) / 2, df=len(correlations_list) - 1)
        margin_of_error = t_value * std_err
        
        lower_bound = mean_corr - margin_of_error
        upper_bound = mean_corr + margin_of_error
        
        return mean_corr, lower_bound, upper_bound
    
    def generate_summary_report(self, model_results, criteria_results, inter_criteria_results):
        """Generate a comprehensive summary report with confidence intervals"""
        print("\nGenerating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("CORRELATION ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model-specific summary with confidence intervals
        report.append("1. MODEL-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        if not model_results.empty:
            # Group by model and calculate confidence intervals
            model_groups = model_results.groupby('model')
            report.append("Average correlation by model (blind vs guided recommendations):")
            for model, group in model_groups:
                correlations = group['correlation'].tolist()
                mean_corr, lower_bound, upper_bound = self.calculate_confidence_intervals(correlations)
                if mean_corr is not None:
                    report.append(f"  {model}: {mean_corr:.3f} (95% CI: [{lower_bound:.3f}, {upper_bound:.3f}], n={len(correlations)})")
                else:
                    report.append(f"  {model}: {mean_corr:.3f} (n={len(correlations)})")
            
            # Criteria-specific confidence intervals
            criteria_groups = model_results.groupby('criteria')
            report.append("\nAverage correlation by ranking criteria:")
            for criteria, group in criteria_groups.head(10):
                correlations = group['correlation'].tolist()
                mean_corr, lower_bound, upper_bound = self.calculate_confidence_intervals(correlations)
                if mean_corr is not None:
                    report.append(f"  {criteria}: {mean_corr:.3f} (95% CI: [{lower_bound:.3f}, {upper_bound:.3f}], n={len(correlations)})")
                else:
                    report.append(f"  {criteria}: {mean_corr:.3f} (n={len(correlations)})")
        report.append("")
        
        # Criteria-specific summary with confidence intervals
        report.append("2. CRITERIA-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        if not criteria_results.empty:
            criteria_model_groups = criteria_results.groupby('criteria')
            report.append("Best performing criteria (highest average correlation):")
            for criteria, group in criteria_model_groups.head(5):
                correlations = group['correlation'].tolist()
                mean_corr, lower_bound, upper_bound = self.calculate_confidence_intervals(correlations)
                if mean_corr is not None:
                    report.append(f"  {criteria}: {mean_corr:.3f} (95% CI: [{lower_bound:.3f}, {upper_bound:.3f}], n={len(correlations)})")
                else:
                    report.append(f"  {criteria}: {mean_corr:.3f} (n={len(correlations)})")
            
            model_criteria_groups = criteria_results.groupby('model')
            report.append("\nBest performing models (highest average correlation):")
            for model, group in model_criteria_groups.head(5):
                correlations = group['correlation'].tolist()
                mean_corr, lower_bound, upper_bound = self.calculate_confidence_intervals(correlations)
                if mean_corr is not None:
                    report.append(f"  {model}: {mean_corr:.3f} (95% CI: [{lower_bound:.3f}, {upper_bound:.3f}], n={len(correlations)})")
                else:
                    report.append(f"  {model}: {mean_corr:.3f} (n={len(correlations)})")
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
                        corr = inter_criteria_results.loc[criteria1, criteria2]
                        similar_pairs.append((criteria1, criteria2, corr))
            
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            report.append("Most similar ranking criteria pairs:")
            for criteria1, criteria2, corr in similar_pairs[:10]:
                report.append(f"  {criteria1} ↔ {criteria2}: {corr:.3f}")
        report.append("")
        
        # Key insights with confidence information
        report.append("4. KEY INSIGHTS")
        report.append("-" * 40)
        if not model_results.empty:
            overall_correlations = model_results['correlation'].tolist()
            overall_avg, overall_lower, overall_upper = self.calculate_confidence_intervals(overall_correlations)
            
            if overall_avg is not None:
                report.append(f"• Overall average correlation: {overall_avg:.3f} (95% CI: [{overall_lower:.3f}, {overall_upper:.3f}], n={len(overall_correlations)})")
            else:
                report.append(f"• Overall average correlation: {overall_avg:.3f} (n={len(overall_correlations)})")
            
            if overall_avg > 0.7:
                report.append("• High correlation suggests LLMs are consistent in their recommendations")
            elif overall_avg > 0.4:
                report.append("• Moderate correlation suggests some consistency with room for improvement")
            else:
                report.append("• Low correlation suggests significant differences between blind and guided recommendations")
            
            # Add confidence interval interpretation
            if overall_lower is not None and overall_upper is not None:
                if overall_lower > 0.7:
                    report.append("• Confidence interval suggests high correlation is statistically significant")
                elif overall_upper < 0.4:
                    report.append("• Confidence interval suggests low correlation is statistically significant")
                else:
                    report.append("• Confidence interval spans multiple correlation levels, suggesting mixed results")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nDetailed report saved to: {self.output_dir / 'correlation_analysis_report.txt'}")
    
    def run_analysis(self):
        """Run the complete correlation analysis"""
        print("Starting correlation analysis...")
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
        
        print("\nCorrelation analysis completed!")
        print(f"All results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze correlations between blind and guided recommendations')
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
    analyzer = CorrelationAnalyzer(args.blind_recommendations, args.guided_recommendations)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
