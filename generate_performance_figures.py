#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-quality figures visualizing the impact of model configuration 
parameters (sequence length, batch size, number of neurons) on computational 
performance metrics (step time and memory consumption).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import argparse
from scipy.stats import sem

# Set publication-ready aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Helvetica'],
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.labelpad': 8,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fontsize': 9,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'figure.titleweight': 'normal'  # Set title weight to normal instead of bold
})

# Configure seaborn style
sns.set_style("ticks")
sns.set_context("paper")

def load_training_data(csv_path):
    """
    Load real training statistics data from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing training statistics
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the loaded training data with error values
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Convert string columns to numeric if needed
        numeric_cols = ['n_neurons', 'batch_size', 'seq_len', 
                        'mean_step_time', 'sem_step_time', 
                        'mean_gpu_memory', 'sem_gpu_memory']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # If we have SEM values already, use them directly
        if 'sem_step_time' in df.columns and 'sem_gpu_memory' in df.columns:
            # Data already has error values
            pass
        else:
            # Group by configuration parameters and calculate statistics
            print("Computing statistics from raw data...")
            grouped = df.groupby(['sim_name', 'n_neurons', 'batch_size', 'seq_len']).agg({
                'mean_step_time': ['mean', 'sem'],
                'mean_gpu_memory': ['mean', 'sem']
            }).reset_index()
            
            # Flatten multi-level column names
            grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
            df = grouped
            
        # Print some statistics about the dataset
        print(f"Unique neuron counts: {sorted(df['n_neurons'].unique())}")
        print(f"Unique batch sizes: {sorted(df['batch_size'].unique())}")
        print(f"Unique sequence lengths: {sorted(df['seq_len'].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['n_neurons', 'batch_size', 'seq_len', 
                                    'mean_step_time', 'sem_step_time',
                                    'mean_gpu_memory', 'sem_gpu_memory'])

def generate_synthetic_training_stats(seq_lens, batch_sizes, neuron_counts, n_samples_per_config=8, output_csv=None):
    """
    Generate synthetic training statistics data that matches the format of training_statistics.csv.
    
    Parameters:
    -----------
    seq_lens : list
        List of sequence lengths to simulate
    batch_sizes : list
        List of batch sizes to simulate
    neuron_counts : list
        List of neuron counts to simulate
    n_samples_per_config : int
        Number of samples to generate for each configuration
    output_csv : str, optional
        Path to save the generated synthetic data as CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic data in the same format as training_statistics.csv
    """
    # Generate raw sample data for each configuration
    raw_data = []
    
    # Define factors that influence time and memory
    seq_len_factor = 2.5e-3  # ms per timestep
    batch_factor = 0.9
    neuron_factor = 1.5e-5
    mem_seq_factor = 2.0e-3
    mem_batch_factor = 0.85
    mem_neuron_factor = 2.0e-5
    
    # Base values
    base_step_time = 0.15  # seconds
    base_memory = 2.0      # GB
    
    # Function to generate simulation names like "b_asf4"
    def generate_sim_name():
        letters = 'abcdefghijklmnopqrstuvwxyz'
        numbers = '0123456789'
        return f"b_{np.random.choice(list(letters))}{np.random.choice(list(letters))}{np.random.choice(list(letters))}{np.random.choice(list(numbers))}"
    
    # Generate data for each configuration
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for neuron_count in neuron_counts:
                # Create a simulation name for this configuration
                sim_name = generate_sim_name()
                
                # Generate multiple samples for this configuration
                step_times = []
                memory_usages = []
                
                for _ in range(n_samples_per_config):
                    # Add noise/variance to make the data look realistic
                    noise_time = np.random.normal(1.0, 0.1)
                    noise_mem = np.random.normal(1.0, 0.05)
                    
                    # Calculate step time based on parameters with some non-linearity
                    step_time = base_step_time + \
                                seq_len * seq_len_factor * noise_time + \
                                batch_size**batch_factor * noise_time * 0.025 + \
                                (neuron_count * neuron_factor)**1.05 * noise_time
                    
                    # Calculate memory consumption
                    memory = base_memory + \
                             seq_len * mem_seq_factor * noise_mem + \
                             batch_size**mem_batch_factor * noise_mem * 0.15 + \
                             neuron_count * mem_neuron_factor * noise_mem
                    
                    # Ensure values are realistic
                    step_time = max(0.1, step_time)
                    memory = max(1.0, memory)
                    
                    step_times.append(step_time)
                    memory_usages.append(memory)
                
                # Calculate mean and standard error for this configuration
                mean_step_time = np.mean(step_times)
                sem_step_time = np.std(step_times) / np.sqrt(len(step_times))
                mean_gpu_memory = np.mean(memory_usages)
                sem_gpu_memory = np.std(memory_usages) / np.sqrt(len(memory_usages))
                
                # Add to the dataset
                raw_data.append({
                    'sim_name': sim_name,
                    'n_neurons': neuron_count,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'mean_step_time': mean_step_time,
                    'sem_step_time': sem_step_time,
                    'mean_gpu_memory': mean_gpu_memory,
                    'sem_gpu_memory': sem_gpu_memory
                })
    
    # Create DataFrame
    df = pd.DataFrame(raw_data)
    
    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Synthetic training statistics saved to: {output_csv}")
    
    return df

def create_comprehensive_figure(df, output_dir='figures/performance_summary'):
    """
    Create a single comprehensive figure that summarizes the key insights of model performance
    for scientific publication.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the performance data
    output_dir : str
        Directory to save the generated figure
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a cohesive color palette
    palette = sns.color_palette("viridis", 8)
    
    # Create figure with 4 panels in a 2x2 arrangement
    fig = plt.figure(figsize=(10, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.4)
    
    # =========================================================================
    # Panel A: Step Time vs Sequence Length for different neuron counts
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Use a fixed batch size (smallest one)
    batch_size = min(df['batch_size'].unique())
    
    # Prepare the data
    seq_len_data = df[df['batch_size'] == batch_size].groupby(['seq_len', 'neurons']).agg(
        step_time=('step_time', 'mean'),
        step_time_err=('step_time_sem', 'mean')
    ).reset_index()
    
    # Sort the neuron counts for better visualization
    unique_neurons = sorted(df['neurons'].unique())
    
    # Select a subset of neuron counts to avoid cluttering
    if len(unique_neurons) > 4:
        # Choose a representative subset: smallest, largest, and 1-2 in between
        if len(unique_neurons) <= 6:
            selected_neurons = unique_neurons
        else:
            selected_neurons = [
                unique_neurons[0],  # smallest
                unique_neurons[len(unique_neurons)//3],  # ~33rd percentile
                unique_neurons[2*len(unique_neurons)//3],  # ~66th percentile
                unique_neurons[-1]  # largest
            ]
    else:
        selected_neurons = unique_neurons
    
    # Plot step time vs sequence length for selected neuron counts
    for i, neurons in enumerate(selected_neurons):
        subset = seq_len_data[seq_len_data['neurons'] == neurons]
        ax1.plot(subset['seq_len'] * 2, subset['step_time'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax1.fill_between(subset['seq_len'] * 2, 
                        subset['step_time'] - subset['step_time_err'],
                        subset['step_time'] + subset['step_time_err'],
                        alpha=0.2, color=palette[i])
    
    ax1.set_xlabel('Sequence Length (timesteps)')
    ax1.set_ylabel('Step Time (s)')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('A) Step Time vs Sequence Length', fontweight='normal', loc='left')
    
    # Add batch size info to the title
    ax1.text(0.5, 1.03, f"Batch Size = {batch_size}", transform=ax1.transAxes,
            ha='center', va='bottom', fontsize=10, style='italic')
    
    # =========================================================================
    # Panel B: Step Time vs Number of Neurons for different sequence lengths
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for different sequence lengths
    neuron_data = df[df['batch_size'] == batch_size].groupby(['neurons', 'seq_len']).agg(
        step_time=('step_time', 'mean'),
        step_time_err=('step_time_sem', 'mean')
    ).reset_index()
    
    # Sort the sequence lengths
    unique_seq_lens = sorted(df['seq_len'].unique())
    
    # Select a subset of sequence lengths if there are too many
    if len(unique_seq_lens) > 4:
        # Choose a representative subset
        selected_seq_lens = [
            unique_seq_lens[0],  # smallest
            unique_seq_lens[len(unique_seq_lens)//2],  # middle
            unique_seq_lens[-1]  # largest
        ]
    else:
        selected_seq_lens = unique_seq_lens
    
    # Plot step time vs number of neurons for selected sequence lengths
    for i, seq_len in enumerate(selected_seq_lens):
        subset = neuron_data[neuron_data['seq_len'] == seq_len]
        ax2.plot(subset['neurons'], subset['step_time'], 
                marker='o', markersize=5, label=f"Seq Len {seq_len * 2}", 
                color=palette[i+4])  # Use different colors than panel A
        ax2.fill_between(subset['neurons'], 
                        subset['step_time'] - subset['step_time_err'],
                        subset['step_time'] + subset['step_time_err'],
                        alpha=0.2, color=palette[i+4])
    
    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('Step Time (s)')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('B) Step Time vs Number of Neurons', fontweight='normal', loc='left')
    
    # Add batch size info to the title
    ax2.text(0.5, 1.03, f"Batch Size = {batch_size}", transform=ax2.transAxes,
            ha='center', va='bottom', fontsize=10, style='italic')
    
    # =========================================================================
    # Panel C: Memory Usage Heatmap
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Prepare data for heatmap - use all available neuron counts and sequence lengths
    heatmap_data = df[df['batch_size'] == batch_size].groupby(['seq_len', 'neurons']).agg(
        mean_memory=('memory', 'mean')
    ).reset_index()
    
    # Create pivot table
    memory_pivot = heatmap_data.pivot(index='seq_len', columns='neurons', values='mean_memory')
    
    # Sort index to have lowest seq_len at bottom, largest at top
    memory_pivot = memory_pivot.sort_index(ascending=False)
    
    # Format column labels
    memory_pivot.columns = [f"{int(col):,}" for col in memory_pivot.columns]
    
    # Convert seq_len index to actual timesteps (multiply by 2)
    memory_pivot.index = memory_pivot.index * 2
    
    # Plot heatmap
    sns.heatmap(memory_pivot, annot=True, fmt=".1f", cmap="YlOrRd", 
               ax=ax3, cbar_kws={'label': 'GPU Memory (GB)'})
    
    ax3.set_title('C) GPU Memory Usage (GB)', fontweight='normal', loc='left')
    ax3.set_xlabel('Number of Neurons')
    ax3.set_ylabel('Sequence Length')
    
    # Add batch size info to the title
    ax3.text(0.5, 1.03, f"Batch Size = {batch_size}", transform=ax3.transAxes,
            ha='center', va='bottom', fontsize=10, style='italic')
    
    # =========================================================================
    # Panel D: Performance Scaling Factors
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate scaling factors for key parameters
    scaling_data = []
    
    # For reference points
    baseline_seq_len = min(df['seq_len'].unique())
    baseline_neurons = min(df['neurons'].unique())
    baseline_batch = min(df['batch_size'].unique())
    
    max_seq_len = max(df['seq_len'].unique())
    max_neurons = max(df['neurons'].unique())
    max_batch = max(df['batch_size'].unique())
    
    # Calculate time scaling for sequence length (using middle neuron count)
    middle_neuron_idx = len(unique_neurons) // 2
    middle_neurons = unique_neurons[middle_neuron_idx]
    
    seq_baseline = df[(df['batch_size'] == baseline_batch) & 
                      (df['neurons'] == middle_neurons) & 
                      (df['seq_len'] == baseline_seq_len)]['step_time'].mean()
    
    seq_max = df[(df['batch_size'] == baseline_batch) & 
                 (df['neurons'] == middle_neurons) & 
                 (df['seq_len'] == max_seq_len)]['step_time'].mean()
    
    seq_scaling = seq_max / seq_baseline
    
    # Calculate time scaling for neuron count
    neuron_baseline = df[(df['batch_size'] == baseline_batch) & 
                        (df['neurons'] == baseline_neurons) & 
                        (df['seq_len'] == baseline_seq_len)]['step_time'].mean()
    
    neuron_max = df[(df['batch_size'] == baseline_batch) & 
                   (df['neurons'] == max_neurons) & 
                   (df['seq_len'] == baseline_seq_len)]['step_time'].mean()
    
    neuron_scaling = neuron_max / neuron_baseline
    
    # Calculate time scaling for batch size
    batch_baseline = df[(df['batch_size'] == baseline_batch) & 
                       (df['neurons'] == middle_neurons) & 
                       (df['seq_len'] == baseline_seq_len)]['step_time'].mean()
    
    batch_max = df[(df['batch_size'] == max_batch) & 
                  (df['neurons'] == middle_neurons) & 
                  (df['seq_len'] == baseline_seq_len)]['step_time'].mean()
    
    batch_scaling = batch_max / batch_baseline
    
    # Calculate memory scaling factors
    mem_seq_baseline = df[(df['batch_size'] == baseline_batch) & 
                         (df['neurons'] == middle_neurons) & 
                         (df['seq_len'] == baseline_seq_len)]['memory'].mean()
    
    mem_seq_max = df[(df['batch_size'] == baseline_batch) & 
                    (df['neurons'] == middle_neurons) & 
                    (df['seq_len'] == max_seq_len)]['memory'].mean()
    
    mem_seq_scaling = mem_seq_max / mem_seq_baseline
    
    mem_neuron_baseline = df[(df['batch_size'] == baseline_batch) & 
                           (df['neurons'] == baseline_neurons) & 
                           (df['seq_len'] == baseline_seq_len)]['memory'].mean()
    
    mem_neuron_max = df[(df['batch_size'] == baseline_batch) & 
                      (df['neurons'] == max_neurons) & 
                      (df['seq_len'] == baseline_seq_len)]['memory'].mean()
    
    mem_neuron_scaling = mem_neuron_max / mem_neuron_baseline
    
    mem_batch_baseline = df[(df['batch_size'] == baseline_batch) & 
                          (df['neurons'] == middle_neurons) & 
                          (df['seq_len'] == baseline_seq_len)]['memory'].mean()
    
    mem_batch_max = df[(df['batch_size'] == max_batch) & 
                     (df['neurons'] == middle_neurons) & 
                     (df['seq_len'] == baseline_seq_len)]['memory'].mean()
    
    mem_batch_scaling = mem_batch_max / mem_batch_baseline
    
    # Prepare data for the bar chart
    scaling_categories = ['Sequence\nLength', 'Number of\nNeurons', 'Batch\nSize']
    time_scaling_values = [seq_scaling, neuron_scaling, batch_scaling]
    mem_scaling_values = [mem_seq_scaling, mem_neuron_scaling, mem_batch_scaling]
    
    # Create grouped bar chart
    x = np.arange(len(scaling_categories))
    width = 0.35
    
    # Plot time scaling bars
    bars1 = ax4.bar(x - width/2, time_scaling_values, width, 
                   label='Step Time Scaling', color=palette[0],
                   edgecolor='black', linewidth=1)
    
    # Plot memory scaling bars
    bars2 = ax4.bar(x + width/2, mem_scaling_values, width,
                   label='Memory Scaling', color=palette[4],
                   edgecolor='black', linewidth=1)
    
    # Add bar labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}×',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Customize the plot
    ax4.set_xticks(x)
    ax4.set_xticklabels(scaling_categories)
    ax4.set_ylabel('Scaling Factor (max/min)')
    ax4.set_title('D) Performance Scaling Factors', fontweight='normal', loc='left')
    ax4.legend(loc='upper right', frameon=True)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add note about scaling factors
    scaling_note = (f"Max/min scaling factors comparing:\n"
                   f"Seq Len: {max_seq_len*2} vs {baseline_seq_len*2} timesteps\n"
                   f"Neurons: {max_neurons:,} vs {baseline_neurons:,}\n"
                   f"Batch Size: {max_batch} vs {baseline_batch}")
    
    ax4.text(0.5, -0.25, scaling_note, transform=ax4.transAxes,
            ha='center', va='top', fontsize=8, style='italic')
    
    # Add a main title for the entire figure
    fig.suptitle('Neural Model Performance Characteristics', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add a figure caption
    caption = ("Figure: Performance analysis of the neural simulation model showing how computational "
              "requirements scale with different parameters. Panels A and B show the relationship between "
              "step time and sequence length/neuron count. Panel C visualizes memory usage across different "
              "configurations. Panel D summarizes the scaling factors for the three key parameters.")
    
    fig.text(0.5, 0.01, caption, wrap=True, ha='center', va='bottom', 
            fontsize=9, style='italic')
    
    # Adjust layout and save
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neural_model_performance_summary.png'), 
               dpi=300, bbox_inches='tight')
    
    print(f"Comprehensive summary figure saved to {output_dir}")
    plt.close()

def plot_performance_figures(df, output_dir='figures/performance'):
    """
    Create publication-quality figures visualizing the relationship between
    model parameters and performance metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the performance data
    output_dir : str
        Directory to save the generated figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a cohesive color palette with enough colors for all neuron counts and sequence lengths
    palette = sns.color_palette("viridis", max(len(df['neurons'].unique()), len(df['seq_len'].unique())))
    
    # =========================================================================
    # FIGURE 1: Sequence Length Impact
    # =========================================================================
    
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[3, 1])
    
    # Panel A: Step Time vs Sequence Length
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare the data (average over samples)
    seq_len_data = df[df['batch_size'] == 1].groupby(['seq_len', 'neurons']).agg(
        mean_step_time=('step_time', 'mean'),
        sem_step_time=('step_time', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        subset = seq_len_data[seq_len_data['neurons'] == neurons]
        # Multiply seq_len by 2 to account for two phases per training step
        ax1.plot(subset['seq_len'] * 2, subset['mean_step_time'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax1.fill_between(subset['seq_len'] * 2, 
                        subset['mean_step_time'] - subset['sem_step_time'],
                        subset['mean_step_time'] + subset['sem_step_time'],
                        alpha=0.2, color=palette[i])
    
    # Set x-ticks to only show values present in the data
    # ax1.set_xticks(sorted(df['seq_len'].unique()))
    
    ax1.set_xlabel('Sequence Length (timesteps)')
    ax1.set_ylabel('Step Time (s)')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('A) Impact of Sequence Length on Step Time', fontweight='normal', loc='left')
    
    # Panel B: Memory vs Sequence Length
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    # Prepare the data
    seq_len_mem_data = df[df['batch_size'] == 1].groupby(['seq_len', 'neurons']).agg(
        mean_memory=('memory', 'mean'),
        sem_memory=('memory', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        subset = seq_len_mem_data[seq_len_mem_data['neurons'] == neurons]
        ax2.plot(subset['seq_len'] * 2, subset['mean_memory'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax2.fill_between(subset['seq_len'] * 2, 
                        subset['mean_memory'] - subset['sem_memory'],
                        subset['mean_memory'] + subset['sem_memory'],
                        alpha=0.2, color=palette[i])
    
    ax2.set_xlabel('Sequence Length (timesteps)')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('C) Impact of Sequence Length on Memory Usage', fontweight='normal', loc='left')
    
    # Panel C: Bar chart showing time scaling factor (normalized to smallest seq_len)
    ax3 = fig.add_subplot(gs[0, 1])
    
    # Calculate scaling factors for sequence length 
    scaling_data = []
    baseline_seq_len = min(df['seq_len'].unique())
    
    # Compute scaling factors (average across neuron counts)
    for neurons in sorted(df['neurons'].unique()):
        neuron_data = df[(df['batch_size'] == 1) & (df['neurons'] == neurons)]
        baseline_time = neuron_data[neuron_data['seq_len'] == baseline_seq_len]['step_time'].mean()
        
        for seq_len in sorted(df['seq_len'].unique()):
            if seq_len != baseline_seq_len:
                current_time = neuron_data[neuron_data['seq_len'] == seq_len]['step_time'].mean()
                scaling_factor = current_time / baseline_time
                scaling_data.append({
                    'neurons': neurons,
                    'seq_len': seq_len,
                    'scaling_factor': scaling_factor
                })
    
    scaling_df = pd.DataFrame(scaling_data)
    
    # Compute average scaling factors across neuron counts
    avg_scaling = scaling_df.groupby('seq_len')['scaling_factor'].mean().reset_index()
    sem_scaling = scaling_df.groupby('seq_len')['scaling_factor'].sem().reset_index()
    
    # Plot scaling factors as bar chart
    bars = ax3.bar(avg_scaling['seq_len'].astype(str), avg_scaling['scaling_factor'], 
                  yerr=sem_scaling['scaling_factor'], capsize=4,
                  color=palette[1], edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time Scaling Factor')
    ax3.set_title('B) Time Scaling', fontweight='normal', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Annoate bars with values
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Panel D: Memory scaling factors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate scaling factors for memory
    mem_scaling_data = []
    
    # Compute scaling factors
    for neurons in sorted(df['neurons'].unique()):
        neuron_data = df[(df['batch_size'] == 1) & (df['neurons'] == neurons)]
        baseline_mem = neuron_data[neuron_data['seq_len'] == baseline_seq_len]['memory'].mean()
        
        for seq_len in sorted(df['seq_len'].unique()):
            if seq_len != baseline_seq_len:
                current_mem = neuron_data[neuron_data['seq_len'] == seq_len]['memory'].mean()
                scaling_factor = current_mem / baseline_mem
                mem_scaling_data.append({
                    'neurons': neurons,
                    'seq_len': seq_len,
                    'scaling_factor': scaling_factor
                })
    
    mem_scaling_df = pd.DataFrame(mem_scaling_data)
    
    # Compute average scaling factors
    avg_mem_scaling = mem_scaling_df.groupby('seq_len')['scaling_factor'].mean().reset_index()
    sem_mem_scaling = mem_scaling_df.groupby('seq_len')['scaling_factor'].sem().reset_index()
    
    # Plot memory scaling factors
    bars = ax4.bar(avg_mem_scaling['seq_len'].astype(str), avg_mem_scaling['scaling_factor'], 
                  yerr=sem_mem_scaling['scaling_factor'], capsize=4,
                  color=palette[2], edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Memory Scaling Factor')
    ax4.set_title('D) Memory Scaling', fontweight='normal', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Add legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.savefig(os.path.join(output_dir, 'sequence_length_impact.png'), dpi=300)
    # plt.savefig(os.path.join(output_dir, 'sequence_length_impact.pdf'))
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Batch Size Impact
    # =========================================================================
    
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[3, 1])
    
    # Panel A: Step Time vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Use the middle sequence length for better visualization
    middle_seq_len = sorted(df['seq_len'].unique())[1]  # e.g., 500
    
    # Prepare the data
    batch_data = df[df['seq_len'] == middle_seq_len].groupby(['batch_size', 'neurons']).agg(
        mean_step_time=('step_time', 'mean'),
        sem_step_time=('step_time', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        subset = batch_data[batch_data['neurons'] == neurons]
        ax1.plot(subset['batch_size'], subset['mean_step_time'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons (Seq Len {middle_seq_len * 2})", 
                color=palette[i])
        ax1.fill_between(subset['batch_size'], 
                        subset['mean_step_time'] - subset['sem_step_time'],
                        subset['mean_step_time'] + subset['sem_step_time'],
                        alpha=0.2, color=palette[i])
    
    # Set x-ticks to only show values present in the data
    # ax1.set_xticks(sorted(df['batch_size'].unique()))
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Step Time (s)')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('A) Impact of Batch Size on Step Time', fontweight='normal', loc='left')
    
    # Panel B: Memory vs Batch Size
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    # Prepare data
    batch_mem_data = df[df['seq_len'] == middle_seq_len].groupby(['batch_size', 'neurons']).agg(
        mean_memory=('memory', 'mean'),
        sem_memory=('memory', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        subset = batch_mem_data[batch_mem_data['neurons'] == neurons]
        ax2.plot(subset['batch_size'], subset['mean_memory'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax2.fill_between(subset['batch_size'], 
                        subset['mean_memory'] - subset['sem_memory'],
                        subset['mean_memory'] + subset['sem_memory'],
                        alpha=0.2, color=palette[i])
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('C) Impact of Batch Size on Memory Usage', fontweight='normal', loc='left')
    
    # Panel C: Bar chart showing time scaling factor
    ax3 = fig.add_subplot(gs[0, 1])
    
    # Calculate scaling factors for batch size
    batch_scaling_data = []
    baseline_batch = min(df['batch_size'].unique())
    
    for neurons in sorted(df['neurons'].unique()):
        neuron_data = df[(df['seq_len'] == middle_seq_len) & (df['neurons'] == neurons)]
        baseline_time = neuron_data[neuron_data['batch_size'] == baseline_batch]['step_time'].mean()
        
        for batch_size in sorted(df['batch_size'].unique()):
            if batch_size != baseline_batch:
                current_time = neuron_data[neuron_data['batch_size'] == batch_size]['step_time'].mean()
                scaling_factor = current_time / baseline_time
                batch_scaling_data.append({
                    'neurons': neurons,
                    'batch_size': batch_size,
                    'scaling_factor': scaling_factor
                })
    
    batch_scaling_df = pd.DataFrame(batch_scaling_data)
    
    # Compute average scaling factors
    avg_batch_scaling = batch_scaling_df.groupby('batch_size')['scaling_factor'].mean().reset_index()
    sem_batch_scaling = batch_scaling_df.groupby('batch_size')['scaling_factor'].sem().reset_index()
    
    # Plot scaling factors
    bars = ax3.bar(avg_batch_scaling['batch_size'].astype(str), 
                  avg_batch_scaling['scaling_factor'],
                  yerr=sem_batch_scaling['scaling_factor'], 
                  capsize=4, color=palette[1], 
                  edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time Scaling Factor')
    ax3.set_title('B) Time Scaling', fontweight='normal', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Panel D: Memory scaling factors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate scaling factors for memory
    batch_mem_scaling_data = []
    
    for neurons in sorted(df['neurons'].unique()):
        neuron_data = df[(df['seq_len'] == middle_seq_len) & (df['neurons'] == neurons)]
        baseline_mem = neuron_data[neuron_data['batch_size'] == baseline_batch]['memory'].mean()
        
        for batch_size in sorted(df['batch_size'].unique()):
            if batch_size != baseline_batch:
                current_mem = neuron_data[neuron_data['batch_size'] == batch_size]['memory'].mean()
                scaling_factor = current_mem / baseline_mem
                batch_mem_scaling_data.append({
                    'neurons': neurons,
                    'batch_size': batch_size,
                    'scaling_factor': scaling_factor
                })
    
    batch_mem_scaling_df = pd.DataFrame(batch_mem_scaling_data)
    
    # Compute average scaling factors
    avg_batch_mem_scaling = batch_mem_scaling_df.groupby('batch_size')['scaling_factor'].mean().reset_index()
    sem_batch_mem_scaling = batch_mem_scaling_df.groupby('batch_size')['scaling_factor'].sem().reset_index()
    
    # Plot memory scaling factors
    bars = ax4.bar(avg_batch_mem_scaling['batch_size'].astype(str), 
                  avg_batch_mem_scaling['scaling_factor'],
                  yerr=sem_batch_mem_scaling['scaling_factor'], 
                  capsize=4, color=palette[2], 
                  edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Memory Scaling Factor')
    ax4.set_title('D) Memory Scaling', fontweight='normal', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.savefig(os.path.join(output_dir, 'batch_size_impact.png'), dpi=300)
    # plt.savefig(os.path.join(output_dir, 'batch_size_impact.pdf'))
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Neuron Count Impact
    # =========================================================================
    
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[3, 1])
    
    # Panel A: Step Time vs Neuron Count
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare the data for different sequence lengths
    neuron_data = df[df['batch_size'] == 1].groupby(['neurons', 'seq_len']).agg(
        mean_step_time=('step_time', 'mean'),
        sem_step_time=('step_time', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, seq_len in enumerate(sorted(df['seq_len'].unique())):
        subset = neuron_data[neuron_data['seq_len'] == seq_len]
        ax1.plot(subset['neurons'], subset['mean_step_time'], 
                marker='o', markersize=5, label=f"Seq Len {seq_len * 2}", 
                color=palette[i])
        ax1.fill_between(subset['neurons'], 
                        subset['mean_step_time'] - subset['sem_step_time'],
                        subset['mean_step_time'] + subset['sem_step_time'],
                        alpha=0.2, color=palette[i])
    
    # Set x-ticks to only show values present in the data
    # ax1.set_xticks(sorted(df['neurons'].unique()))
    
    ax1.set_xlabel('Number of Neurons')
    ax1.set_ylabel('Step Time (s)')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('A) Impact of Neuron Count on Step Time', fontweight='normal', loc='left')
    
    # Panel B: Memory vs Neuron Count
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    # Prepare data
    neuron_mem_data = df[df['batch_size'] == 1].groupby(['neurons', 'seq_len']).agg(
        mean_memory=('memory', 'mean'),
        sem_memory=('memory', 'sem')
    ).reset_index()
    
    # Create line plot with error bands
    for i, seq_len in enumerate(sorted(df['seq_len'].unique())):
        subset = neuron_mem_data[neuron_mem_data['seq_len'] == seq_len]
        ax2.plot(subset['neurons'], subset['mean_memory'], 
                marker='o', markersize=5, label=f"Seq Len {seq_len * 2}", 
                color=palette[i])
        ax2.fill_between(subset['neurons'], 
                        subset['mean_memory'] - subset['sem_memory'],
                        subset['mean_memory'] + subset['sem_memory'],
                        alpha=0.2, color=palette[i])
    
    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('C) Impact of Neuron Count on Memory Usage', fontweight='normal', loc='left')
    
    # Panel C: Bar chart showing time scaling factor
    ax3 = fig.add_subplot(gs[0, 1])
    
    # Calculate scaling factors for neuron count
    neuron_scaling_data = []
    baseline_neurons = min(df['neurons'].unique())
    
    for seq_len in sorted(df['seq_len'].unique()):
        seq_data = df[(df['batch_size'] == 1) & (df['seq_len'] == seq_len)]
        baseline_time = seq_data[seq_data['neurons'] == baseline_neurons]['step_time'].mean()
        
        for neurons in sorted(df['neurons'].unique()):
            if neurons != baseline_neurons:
                current_time = seq_data[seq_data['neurons'] == neurons]['step_time'].mean()
                scaling_factor = current_time / baseline_time
                neuron_scaling_data.append({
                    'seq_len': seq_len,
                    'neurons': neurons,
                    'scaling_factor': scaling_factor
                })
    
    neuron_scaling_df = pd.DataFrame(neuron_scaling_data)
    
    # Get the largest neuron count for visualization
    largest_neuron_count = max(df['neurons'].unique())
    
    # Filter to show scaling for the largest neuron count
    largest_scaling = neuron_scaling_df[neuron_scaling_df['neurons'] == largest_neuron_count]
    
    # Plot scaling factors
    bars = ax3.bar(largest_scaling['seq_len'].astype(str), 
                  largest_scaling['scaling_factor'],
                  color=palette[1], edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel(f'Time Scaling\n({largest_neuron_count:,} vs {baseline_neurons:,} neurons)')
    ax3.set_title('B) Time Scaling', fontweight='normal', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Panel D: Memory scaling factors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate scaling factors for memory
    neuron_mem_scaling_data = []
    
    for seq_len in sorted(df['seq_len'].unique()):
        seq_data = df[(df['batch_size'] == 1) & (df['seq_len'] == seq_len)]
        baseline_mem = seq_data[seq_data['neurons'] == baseline_neurons]['memory'].mean()
        
        for neurons in sorted(df['neurons'].unique()):
            if neurons != baseline_neurons:
                current_mem = seq_data[seq_data['neurons'] == neurons]['memory'].mean()
                scaling_factor = current_mem / baseline_mem
                neuron_mem_scaling_data.append({
                    'seq_len': seq_len,
                    'neurons': neurons,
                    'scaling_factor': scaling_factor
                })
    
    neuron_mem_scaling_df = pd.DataFrame(neuron_mem_scaling_data)
    
    # Filter to show scaling for the largest neuron count
    largest_mem_scaling = neuron_mem_scaling_df[neuron_mem_scaling_df['neurons'] == largest_neuron_count]
    
    # Plot memory scaling factors
    bars = ax4.bar(largest_mem_scaling['seq_len'].astype(str), 
                  largest_mem_scaling['scaling_factor'],
                  color=palette[2], edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel(f'Memory Scaling\n({largest_neuron_count:,} vs {baseline_neurons:,} neurons)')
    ax4.set_title('D) Memory Scaling', fontweight='normal', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=len(df['seq_len'].unique()))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.savefig(os.path.join(output_dir, 'neuron_count_impact.png'), dpi=300)
    # plt.savefig(os.path.join(output_dir, 'neuron_count_impact.pdf'))
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Summary Heatmaps
    # =========================================================================
    
    # This visualization will provide a comprehensive overview of performance
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use a middle value for batch size
    middle_batch = sorted(df['batch_size'].unique())[0]  # e.g., batch_size=2
    
    # Prepare data for heatmaps
    heatmap_data = df[df['batch_size'] == middle_batch].groupby(['seq_len', 'neurons']).agg(
        mean_step_time=('step_time', 'mean'),
        mean_memory=('memory', 'mean')
    ).reset_index()
    
    # Create pivot tables
    time_pivot = heatmap_data.pivot(index='seq_len', columns='neurons', values='mean_step_time')
    memory_pivot = heatmap_data.pivot(index='seq_len', columns='neurons', values='mean_memory')
    
    # Sort index to have lowest seq_len at bottom, largest at top
    time_pivot = time_pivot.sort_index(ascending=False)
    memory_pivot = memory_pivot.sort_index(ascending=False)
    
    # Format column labels
    time_pivot.columns = [f"{int(col):,}" for col in time_pivot.columns]
    memory_pivot.columns = [f"{int(col):,}" for col in memory_pivot.columns]
    
    # Multiply seq_len values by 2 for displaying in heatmap
    time_pivot.index = time_pivot.index * 2
    memory_pivot.index = memory_pivot.index * 2
    
    # Plot heatmaps
    sns.heatmap(time_pivot, annot=True, fmt=".2f", cmap="YlGnBu", 
               ax=axs[0], cbar_kws={'label': 'Step Time (s)'})
    sns.heatmap(memory_pivot, annot=True, fmt=".1f", cmap="YlOrRd", 
               ax=axs[1], cbar_kws={'label': 'GPU Memory (GB)'})
    
    # Set titles and labels with normal weight (not bold)
    axs[0].set_title(f'A) Step Time (s) for Batch Size = {middle_batch}', 
                    fontweight='normal', loc='left')
    axs[1].set_title(f'B) GPU Memory (GB) for Batch Size = {middle_batch}', 
                    fontweight='normal', loc='left')
    
    for ax in axs:
        ax.set_xlabel('Number of Neurons')
        ax.set_ylabel('Sequence Length')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmaps.png'), dpi=300)
#     plt.savefig(os.path.join(output_dir, 'performance_heatmaps.pdf'))
    plt.close()


def main():
    """
    Main function to load real training data and create performance analysis figures.
    """
    # Path to the real training data
    csv_path = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/training_statistics.csv'
    
    # Create output directory
    output_dir = 'figures/performance_real'
    os.makedirs(output_dir, exist_ok=True)
    
    # # Load real training data
    # print("Loading real training data...")
    # df = load_training_data(csv_path)
        
    # # Rename columns to match the expected format in the plot functions
    # df = df.rename(columns={
    #     'n_neurons': 'neurons',
    #     'mean_step_time': 'step_time', 
    #     'sem_step_time': 'step_time_sem',
    #     'mean_gpu_memory': 'memory',
    #     'sem_gpu_memory': 'memory_sem'
    # })
    
    # # Generate figures
    # print("Creating performance analysis figures...")
    # plot_performance_figures(df, output_dir)
    # print(f"Saved figures to {output_dir}")
    
    # # Generate comprehensive figure for scientific publication
    # print("Creating comprehensive summary figure for scientific publication...")
    # create_comprehensive_figure(df, output_dir + "_summary")
    # print(f"Saved comprehensive summary figure to {output_dir}_summary")

    # Create output directory
    output_dir = 'figures/performance_synthetic'
    os.makedirs(output_dir, exist_ok=True)

    df = generate_synthetic_training_stats(
        seq_lens=[200, 500, 1000, 2000],
        batch_sizes=[1, 2, 4, 6],
        neuron_counts=[1000, 5000, 10000, 20000, 40000, 65871],
        n_samples_per_config=8
    )

    # Rename columns to match the expected format in the plot functions
    df = df.rename(columns={
        'n_neurons': 'neurons',
        'mean_step_time': 'step_time', 
        'sem_step_time': 'step_time_sem',
        'mean_gpu_memory': 'memory',
        'sem_gpu_memory': 'memory_sem'
    })
    
    # Generate figures
    print("Creating performance analysis figures...")
    plot_performance_figures(df, output_dir)
    print(f"Saved figures to {output_dir}")
    
    # Generate comprehensive figure for scientific publication from synthetic data too
    print("Creating comprehensive summary figure for scientific publication (synthetic data)...")
    create_comprehensive_figure(df, output_dir + "_summary")
    print(f"Saved synthetic comprehensive summary figure to {output_dir}_summary")

if __name__ == "__main__":
    main()