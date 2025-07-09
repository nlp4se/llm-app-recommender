import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

# Global array of k values
K_VALUES = [ 20]

def rbo(list1, list2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two lists.
    p is the persistence parameter, giving weight to top-ranked items.
    Returns a value between 0 and 1, where 1 means perfect agreement.
    """
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    
    if not list1 or not list2:
        return 0

    # If lists are identical, return 1
    if list1 == list2:
        return 1.0

    s_len, t_len = len(list1), len(list2)
    max_depth = max(s_len, t_len)
    
    # Calculate agreement at each depth
    agreement = 0.0
    normalization = 0.0
    
    for d in range(1, max_depth + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        agreement_d = len(set1.intersection(set2)) / d
        agreement += agreement_d * (p ** (d-1))
        normalization += p ** (d-1)

    # Normalize by the sum of weights
    return agreement / normalization if normalization > 0 else 0

class RBOAnalyzer:
    def __init__(self, blind_recommendations_path, guided_recommendations_path, k_values=None):
        """
        Initialize the RBO analyzer with two CSV files:
        - blind_recommendations: recommendations without specific ranking criteria
        - guided_recommendations: recommendations with specific ranking criteria
        - k_values: list of k values to analyze
        """
        self.blind_df = pd.read_csv(blind_recommendations_path)
        self.guided_df = pd.read_csv(guided_recommendations_path)
        
        # Set k values
        self.k_values = k_values if k_values is not None else K_VALUES
        
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
        
        # Replace underscores with spaces in ranking criteria names
        self.guided_df['ranking_criteria'] = self.guided_df['ranking_criteria'].str.replace('_', ' ')
        
        # Get unique models, features, and criteria - preserve order from input files
        self.models = self.blind_df['model'].unique()
        self.features = self.blind_df['feature'].unique()
        # FIXED: Use the correct column name 'ranking_criteria' instead of 'prefix'
        self.criteria = self.guided_df['ranking_criteria'].unique()
        
        print(f"Found {len(self.models)} models: {self.models}")
        print(f"Found {len(self.features)} features: {self.features}")
        print(f"Found {len(self.criteria)} ranking criteria: {self.criteria}")
        
        # DEBUG: Print some sample data to understand the structure
        print(f"\nBlind recommendations sample:")
        print(self.blind_df.head(2))
        print(f"\nGuided recommendations sample:")
        print(self.guided_df.head(2))
        
    def get_ranked_lists(self, df, model, feature, k):
        """
        Extracts ranked lists of apps for a given model and feature, truncated to k items.
        """
        model_df = df[(df['model'] == model) & (df['feature'] == feature)]
        # Get ranking columns from '1' to k
        rank_cols = [str(i) for i in range(1, k + 1)]
        # Ensure columns exist in dataframe before trying to access them
        rank_cols_exist = [col for col in rank_cols if col in model_df.columns]
        lists = model_df[rank_cols_exist].values.tolist()
        return [[app for app in l if pd.notna(app) and app != ''] for l in lists]
        
    def model_specific_analysis_for_k(self, k):
        """
        Model-specific analysis: RBO between blind recommendations and each ranking criteria for a specific k value
        """
        print(f"\nPerforming model-specific analysis for k={k}...")
        
        results = []
        
        for model in self.models:
            print(f"Processing model: {model}")
            
            for feature in self.features:
                # Get blind recommendations for this model and feature, truncated to k
                blind_lists = self.get_ranked_lists(self.blind_df, model, feature, k)
                
                if not blind_lists:
                    print(f"  No blind lists found for {model}, {feature}")
                    continue
                
                print(f"  Found {len(blind_lists)} blind lists for {model}, {feature}")
                
                # Calculate RBO for each ranking criteria
                for criteria in self.criteria:
                    # FIXED: Use the correct column name 'ranking_criteria' instead of 'prefix'
                    guided_criteria_df = self.guided_df[
                        (self.guided_df['model'] == model) & 
                        (self.guided_df['feature'] == feature) & 
                        (self.guided_df['ranking_criteria'] == criteria)
                    ]
                    
                    if guided_criteria_df.empty:
                        print(f"    No guided data found for {model}, {feature}, {criteria}")
                        continue
                    
                    guided_criteria_lists = self.get_ranked_lists(guided_criteria_df, model, feature, k)
                    
                    if not guided_criteria_lists:
                        print(f"    No guided lists found for {model}, {feature}, {criteria}")
                        continue
                    
                    print(f"    Found {len(guided_criteria_lists)} guided lists for {model}, {feature}, {criteria}")
                    
                    # Calculate RBO between blind and guided lists
                    rbo_scores = []
                    for blind_list in blind_lists:
                        for guided_list in guided_criteria_lists:
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
                        print(f"      RBO: {avg_rbo:.3f} (n={len(rbo_scores)})")
                    else:
                        print(f"      No valid RBO calculations")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def criteria_specific_analysis_for_k(self, k):
        """
        Criteria-specific analysis: compare how different models perform for each criteria for a specific k value
        """
        print(f"\nPerforming criteria-specific analysis for k={k}...")
        
        results = []
        
        for criteria in self.criteria:
            print(f"Processing criteria: {criteria}")
            
            for feature in self.features:
                # Get guided recommendations for this criteria and feature, truncated to k
                # FIXED: Use the correct column name 'ranking_criteria' instead of 'prefix'
                guided_criteria_df = self.guided_df[
                    (self.guided_df['ranking_criteria'] == criteria) & 
                    (self.guided_df['feature'] == feature)
                ]
                
                if guided_criteria_df.empty:
                    continue
                
                # FIXED: Use the correct model and feature parameters
                guided_criteria_lists = []
                for model in self.models:
                    model_lists = self.get_ranked_lists(guided_criteria_df, model, feature, k)
                    guided_criteria_lists.extend(model_lists)
                
                if not guided_criteria_lists:
                    continue
                
                # Compare each model's blind recommendations with this criteria
                for model in self.models:
                    blind_lists = self.get_ranked_lists(self.blind_df, model, feature, k)
                    
                    if not blind_lists:
                        continue
                    
                    # Get guided lists for this specific model and criteria
                    guided_model_lists = self.get_ranked_lists(guided_criteria_df, model, feature, k)
                    
                    if not guided_model_lists:
                        continue
                    
                    # Calculate RBO
                    rbo_scores = []
                    for blind_list in blind_lists:
                        for guided_list in guided_model_lists:
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
        
        return results_df
    
    def inter_criteria_analysis_for_k(self, k):
        """
        Inter-criteria analysis: find RBO between different ranking criteria for a specific k value
        """
        print(f"\nPerforming inter-criteria analysis for k={k}...")
        
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
                            # Get recommendations for both criteria, truncated to k
                            # FIXED: Use the correct column name 'ranking_criteria' instead of 'prefix'
                            guided1_df = self.guided_df[
                                (self.guided_df['model'] == model) & 
                                (self.guided_df['feature'] == feature) & 
                                (self.guided_df['ranking_criteria'] == criteria1)
                            ]
                            
                            guided2_df = self.guided_df[
                                (self.guided_df['model'] == model) & 
                                (self.guided_df['feature'] == feature) & 
                                (self.guided_df['ranking_criteria'] == criteria2)
                            ]
                            
                            if guided1_df.empty or guided2_df.empty:
                                continue
                            
                            guided1_lists = self.get_ranked_lists(guided1_df, model, feature, k)
                            guided2_lists = self.get_ranked_lists(guided2_df, model, feature, k)
                            
                            if not guided1_lists or not guided2_lists:
                                continue
                            
                            # Calculate RBO between rankings
                            for list1 in guided1_lists:
                                for list2 in guided2_lists:
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
        
        return overall_df, model_results
    
    def plot_model_specific_rbo_for_k(self, results_df, k, output_dir):
        """Create heatmap for model-specific RBO for a specific k value"""
        if results_df.empty:
            print(f"No data for model-specific RBO plot (k={k})")
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
        
        # Create heatmap with YlGnBu color palette
        sns.heatmap(
            pivot_data_with_avg, 
            annot=True, 
            cmap='YlGnBu', 
            vmin=0, vmax=1,
            fmt='.2f',
            cbar_kws={'label': 'RBO Score'},
            ax=ax
        )
        
        ax.set_title(f'Model-Specific RBO: Blind vs Guided Recommendations (k={k})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_specific_rbo_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate summary table with averages
        summary_data = pd.DataFrame({
            'Average RBO': overall_avg
        })
        summary_data.to_csv(output_dir / 'model_averages_summary.csv')
        
        print(f"Model-specific RBO heatmap for k={k} saved to: {output_dir / 'model_specific_rbo_heatmap.png'}")
    
    def plot_criteria_specific_rbo_for_k(self, results_df, k, output_dir):
        """Create box plots for criteria-specific RBO for a specific k value"""
        if results_df.empty:
            print(f"No data for criteria-specific RBO plot (k={k})")
            return
            
        # Create figure with subplots - optimized space usage
        n_criteria = len(results_df['criteria'].unique())
        n_cols = 4
        n_rows = (n_criteria + n_cols - 1) // n_cols
        
        # Optimized figure size - reduced height multiplier
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Set font sizes for better readability (copied from app_consistency.py)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # Modern color palette
        modern_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', 
                        '#3A0CA3', '#4361EE', '#4CC9F0', '#F72585', '#7209B7']
        
        for i, criteria in enumerate(sorted(results_df['criteria'].unique())):
            if i >= len(axes):
                break
                
            ax = axes[i]
            criteria_data = results_df[results_df['criteria'] == criteria]
            
            # Create box plot with modern colors
            box_plot = sns.boxplot(data=criteria_data, x='model', y='rbo', ax=ax, 
                                  palette=modern_colors[:len(criteria_data['model'].unique())])
            
            # Optimized title and labels - removed unnecessary elements
            ax.set_title(f'{criteria}', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('RBO', fontsize=12, fontweight='semibold')
            
            # Optimized tick parameters
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            
            # Set y-axis limits to 0 and 1 for consistent scale
            ax.set_ylim(0, 1)
            
            # Add mean points with optimized size
            means = criteria_data.groupby('model')['rbo'].mean()
            for j, model in enumerate(means.index):
                ax.scatter(j, means[model], color='#FF6B6B', s=60, zorder=5, marker='o', 
                          edgecolors='white', linewidth=1.0)
                # Optimized annotation
                ax.annotate(f'{means[model]:.2f}', 
                           (j, means[model]), 
                           xytext=(0, 10), 
                           textcoords='offset points', 
                           ha='center', 
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(results_df['criteria'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        # Optimized layout
        plt.tight_layout(pad=1.5)
        plt.savefig(output_dir / 'criteria_specific_rbo_boxplot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Criteria-specific RBO boxplot for k={k} saved to: {output_dir / 'criteria_specific_rbo_boxplot.png'}")
    
    def plot_inter_criteria_rbo_by_model_for_k(self, model_results, k, output_dir):
        """Create heatmaps for inter-criteria RBO for each model for a specific k value"""
        if not model_results:
            print(f"No data for model-specific inter-criteria RBO plots (k={k})")
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
            
            ax.set_title(f'Inter-Criteria RBO: {model} (k={k})', fontsize=16, pad=20)
            ax.set_xlabel('Ranking Criteria', fontsize=12)
            ax.set_ylabel('Ranking Criteria', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save individual file for this model
            clean_model_name = model.replace(' ', '_').replace('-', '_').replace('+', 'plus')
            filename = f'inter_criteria_rbo_{clean_model_name}.png'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Inter-criteria RBO heatmap for {model} (k={k}) saved to: {output_dir / filename}")
    
    def plot_inter_criteria_rbo_overall_for_k(self, criteria_df, k, output_dir):
        """Create heatmap for overall inter-criteria RBO for a specific k value"""
        if criteria_df.empty:
            print(f"No data for overall inter-criteria RBO plot (k={k})")
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
        
        ax.set_title(f'Overall Inter-Criteria RBO: Average Across All Models (k={k})', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Ranking Criteria', fontsize=12)
        ax.set_ylabel('Ranking Criteria', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'inter_criteria_rbo_overall.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overall inter-criteria RBO heatmap for k={k} saved to: {output_dir / 'inter_criteria_rbo_overall.png'}")
    
    def create_global_heatmap_grid(self, all_results, k, output_dir):
        """
        Create a global 3x2 grid figure with 5 model-specific heatmaps and 1 overall heatmap.
        The overall heatmap is subtly highlighted.
        """
        if not all_results:
            print(f"No data for global heatmap grid (k={k})")
            return
        
        # Set font sizes for better readability (copied from app_consistency.py)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # Create 3x2 grid
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # Get model-specific results and overall results
        model_results = all_results.get('model_results', {})
        inter_criteria_results = all_results.get('inter_criteria_results', pd.DataFrame())
        
        # Plot first 5 model-specific heatmaps
        model_count = 0
        for i, (model, criteria_df) in enumerate(model_results.items()):
            if model_count >= 5:  # Only plot first 5 models
                break
                
            ax = axes[i]
            
            if not criteria_df.empty:
                # Create heatmap for this model
                sns.heatmap(
                    criteria_df, 
                    annot=True, 
                    cmap='YlGnBu', 
                    vmin=0, vmax=1,
                    fmt='.2f',
                    cbar=False,  # No colorbar for individual plots
                    ax=ax,
                    square=True,
                    annot_kws={'size': 8}  # Smaller annotations for grid
                )
                
                ax.set_title(f'{model}', fontsize=12, pad=10)
                ax.set_xlabel('Ranking Criteria', fontsize=10)
                ax.set_ylabel('Ranking Criteria', fontsize=10)
                
                # Rotate labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            
            model_count += 1
        
        # Plot overall heatmap in the last position (subtly highlighted)
        ax = axes[5]
        if not inter_criteria_results.empty:
            # Create heatmap with subtle highlighting (slightly different color or border)
            heatmap = sns.heatmap(
                inter_criteria_results, 
                annot=True, 
                cmap='YlGnBu', 
                vmin=0, vmax=1,
                fmt='.2f',
                cbar=True,  # Show colorbar for overall plot
                ax=ax,
                square=True,
                annot_kws={'size': 8},
                cbar_kws={'label': 'RBO Score', 'shrink': 0.8}
            )
            
            # Subtle highlighting for overall heatmap
            ax.set_title('OVERALL AVERAGE', fontsize=12, pad=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
            ax.set_xlabel('Ranking Criteria', fontsize=10)
            ax.set_ylabel('Ranking Criteria', fontsize=10)
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # Hide empty subplots if we have fewer than 5 models
        for i in range(model_count, 5):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Inter-Criteria RBO Analysis Grid (k={k})', fontsize=16, y=0.95)
        
        # Optimized layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Save the figure
        plt.savefig(output_dir / 'global_inter_criteria_rbo_grid.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Global inter-criteria RBO grid for k={k} saved to: {output_dir / 'global_inter_criteria_rbo_grid.png'}")
    
    def generate_summary_report_for_k(self, model_results, criteria_results, inter_criteria_results, k, output_dir):
        """Generate a comprehensive summary report for a specific k value"""
        print(f"\nGenerating summary report for k={k}...")
        
        report = []
        report.append("=" * 80)
        report.append(f"RBO ANALYSIS SUMMARY REPORT (k={k})")
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
        with open(output_dir / 'rbo_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Detailed report for k={k} saved to: {output_dir / 'rbo_analysis_report.txt'}")
    
    def analyze_for_k(self, k):
        """
        Run all three types of analysis for a specific k value and return results
        """
        print(f"\nRunning complete analysis for k={k}...")
        
        # Run model-specific analysis
        model_results = self.model_specific_analysis_for_k(k)
        
        # Run criteria-specific analysis  
        criteria_results = self.criteria_specific_analysis_for_k(k)
        
        # Run inter-criteria analysis
        inter_criteria_results, model_inter_results = self.inter_criteria_analysis_for_k(k)
        
        # Create output directory for this k value
        k_output_dir = self.output_dir / f'k{k}'
        k_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots and reports for this k value
        if not model_results.empty:
            self.plot_model_specific_rbo_for_k(model_results, k, k_output_dir)
        
        if not criteria_results.empty:
            self.plot_criteria_specific_rbo_for_k(criteria_results, k, k_output_dir)
        
        if not inter_criteria_results.empty:
            self.plot_inter_criteria_rbo_overall_for_k(inter_criteria_results, k, k_output_dir)
            self.plot_inter_criteria_rbo_by_model_for_k(model_inter_results, k, k_output_dir)
            self.create_global_heatmap_grid({
                'model_results': model_inter_results,
                'inter_criteria_results': inter_criteria_results
            }, k, k_output_dir)
        
        # Generate summary report for this k value
        self.generate_summary_report_for_k(model_results, criteria_results, inter_criteria_results, k, k_output_dir)
        
        return model_results, criteria_results, inter_criteria_results
    
    def run_analysis(self):
        """Run the complete RBO analysis for all k values"""
        print("Starting RBO analysis...")
        print(f"Blind recommendations: {len(self.blind_df)} rows")
        print(f"Guided recommendations: {len(self.guided_df)} rows")
        print(f"Analyzing k values: {self.k_values}")
        
        # Preprocess data
        self.preprocess_data()
        
        # Run analysis for each k value
        all_results = {}
        for k in self.k_values:
            print(f"\n{'='*60}")
            print(f"ANALYZING FOR k={k}")
            print(f"{'='*60}")
            
            model_results, criteria_results, inter_criteria_results = self.analyze_for_k(k)
            all_results[k] = {
                'model_results': model_results,
                'criteria_results': criteria_results,
                'inter_criteria_results': inter_criteria_results
            }
        
        # Generate overall summary report
        self.generate_overall_summary_report(all_results)
        
        print("\nRBO analysis completed!")
        print(f"All results saved to: {self.output_dir}")
        print(f"Individual k-value results saved to subfolders: {[f'k{k}' for k in self.k_values]}")
    
    def generate_overall_summary_report(self, all_results):
        """Generate an overall summary report comparing results across all k values"""
        print("\nGenerating overall summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("OVERALL RBO ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Compare results across k values
        report.append("1. COMPARISON ACROSS K VALUES")
        report.append("-" * 40)
        
        k_comparison = []
        for k, results in all_results.items():
            if not results['model_results'].empty:
                overall_avg = results['model_results']['rbo'].mean()
                k_comparison.append((k, overall_avg))
        
        if k_comparison:
            report.append("Overall average RBO by k value:")
            for k, avg_rbo in sorted(k_comparison):
                report.append(f"  k={k}: {avg_rbo:.3f}")
            
            # Find best and worst k values
            best_k = max(k_comparison, key=lambda x: x[1])
            worst_k = min(k_comparison, key=lambda x: x[1])
            report.append(f"\nBest performing k value: k={best_k[0]} (RBO={best_k[1]:.3f})")
            report.append(f"Worst performing k value: k={worst_k[0]} (RBO={worst_k[1]:.3f})")
        
        report.append("")
        report.append("2. KEY FINDINGS")
        report.append("-" * 40)
        report.append("• Analysis completed for all specified k values")
        report.append("• Individual results saved in k-specific subfolders")
        report.append("• Each subfolder contains model-specific, criteria-specific, and inter-criteria analyses")
        report.append("")
        report.append("=" * 80)
        
        # Save overall report
        report_text = "\n".join(report)
        with open(self.output_dir / 'overall_rbo_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nOverall report saved to: {self.output_dir / 'overall_rbo_analysis_report.txt'}")

def main():
    parser = argparse.ArgumentParser(description='Analyze RBO between blind and guided recommendations')
    parser.add_argument('--blind-recommendations', type=str, 
                       default='data/output/evaluation/app_rankings.csv',
                       help='Path to blind recommendations CSV file')
    parser.add_argument('--guided-recommendations', type=str,
                       default='data/output/evaluation/correlation/app_rankings.csv',
                       help='Path to guided recommendations CSV file')
    parser.add_argument('--k-values', nargs='+', type=int, default=K_VALUES,
                       help='List of k values to analyze (e.g., 3 10 20). Default: 3 10 20')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.blind_recommendations).exists():
        print(f"Error: Blind recommendations file not found: {args.blind_recommendations}")
        return
    
    if not Path(args.guided_recommendations).exists():
        print(f"Error: Guided recommendations file not found: {args.guided_recommendations}")
        return
    
    # Run analysis
    analyzer = RBOAnalyzer(args.blind_recommendations, args.guided_recommendations, args.k_values)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
