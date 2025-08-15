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
        numeric_cols = ['n_neurons', 'n_edges', 'batch_size', 'seq_len', 
                        'mean_step_time', 'sem_step_time', 
                        'mean_gpu_memory', 'sem_gpu_memory',
                        'mean_rate', 'sem_rate']
        
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
            grouped = df.groupby(['sim_name', 'n_neurons', 'n_edges', 'batch_size', 'seq_len', 'mode']).agg({
                'mean_step_time': ['mean', 'sem'],
                'mean_gpu_memory': ['mean', 'sem'],
                'mean_rate': ['mean', 'sem']
            }).reset_index()
            
            # Flatten multi-level column names
            grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
            df = grouped
            
        # Print some statistics about the dataset
        print(f"Unique neuron counts: {sorted(df['n_neurons'].unique())}")
        print(f"Unique batch sizes: {sorted(df['batch_size'].unique())}")
        print(f"Unique sequence lengths: {sorted(df['seq_len'].unique())}")
        if 'mode' in df.columns:
            print(f"Modes: {sorted(df['mode'].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['n_neurons', 'n_edges', 'batch_size', 'seq_len', 
                                    'mean_step_time', 'sem_step_time',
                                    'mean_gpu_memory', 'sem_gpu_memory',
                                    'mean_rate', 'sem_rate', 'mode'])

def generate_synthetic_training_stats(seq_lens, batch_sizes, neuron_counts, n_samples_per_config=8, output_csv=None, include_test_mode=True):
    """
    Generate synthetic training statistics data that matches the format of performance_statistics.csv.
    
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
    include_test_mode : bool, default=True
        Whether to include test mode data in addition to train mode
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic data in the same format as performance_statistics.csv
    """
    # Generate raw sample data for each configuration
    raw_data = []
    
    # Define factors that influence time and memory
    seq_len_factor = 2.5e-3     # ms per timestep
    batch_factor = 0.9
    neuron_factor = 1.5e-5
    mem_seq_factor = 2.0e-3
    mem_batch_factor = 0.85
    mem_neuron_factor = 2.0e-5
    edge_neuron_ratio_base = 100  # Average number of edges per neuron as base value
    edge_neuron_ratio_var = 20    # Variation in edge/neuron ratio
    
    # Base values
    base_step_time = 0.15  # seconds
    base_memory = 2.0      # GB
    
    # Function to generate simulation names like "b_asf4"
    def generate_sim_name():
        letters = 'abcdefghijklmnopqrstuvwxyz'
        numbers = '0123456789'
        return f"b_{np.random.choice(list(letters))}{np.random.choice(list(letters))}{np.random.choice(list(letters))}{np.random.choice(list(numbers))}"
    
    # Modes to generate data for
    modes = ["train", "test"] if include_test_mode else ["train"]
    
    # Generate data for each mode, configuration
    for mode in modes:
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                for neuron_count in neuron_counts:
                    # Create a simulation name for this configuration
                    sim_name = generate_sim_name()
                    
                    # Calculate number of edges based on neuron count with some variation
                    edge_neuron_ratio = edge_neuron_ratio_base + np.random.normal(0, edge_neuron_ratio_var)
                    edge_neuron_ratio = max(10, edge_neuron_ratio)  # Ensure minimum connectivity
                    n_edges = int(neuron_count * edge_neuron_ratio)
                    
                    # Generate multiple samples for this configuration
                    step_times = []
                    memory_usages = []
                    firing_rates = []
                    
                    for _ in range(n_samples_per_config):
                        # Add noise/variance to make the data look realistic
                        noise_time = np.random.normal(1.0, 0.1)
                        noise_mem = np.random.normal(1.0, 0.05)
                        
                        # Calculate step time with mode differences
                        if mode == "train":
                            # Training mode has longer step times due to backprop
                            mode_factor = 1.0
                            step_time = base_step_time + \
                                        seq_len * seq_len_factor * noise_time + \
                                        batch_size**batch_factor * noise_time * 0.025 + \
                                        (neuron_count * neuron_factor)**1.05 * noise_time
                        else:
                            # Test mode is faster (no backprop)
                            mode_factor = 0.6
                            step_time = (base_step_time + \
                                        seq_len * seq_len_factor * noise_time * 0.5 + \
                                        batch_size**batch_factor * noise_time * 0.02 + \
                                        (neuron_count * neuron_factor)**1.05 * noise_time) * mode_factor
                                        
                        # Calculate memory with mode differences
                        if mode == "train":
                            # Training mode uses more memory for gradients
                            mem_mode_factor = 1.0
                            memory = base_memory + \
                                    seq_len * mem_seq_factor * noise_mem + \
                                    batch_size**mem_batch_factor * noise_mem * 0.15 + \
                                    neuron_count * mem_neuron_factor * noise_mem
                        else:
                            # Test mode uses less memory (no gradient storage)
                            mem_mode_factor = 0.8
                            memory = (base_memory + \
                                    seq_len * mem_seq_factor * noise_mem * 0.7 + \
                                    batch_size**mem_batch_factor * noise_mem * 0.1 + \
                                    neuron_count * mem_neuron_factor * noise_mem) * mem_mode_factor
                        
                        # Calculate mean firing rate with realistic values (5-40 Hz)
                        # Larger networks tend to have lower average firing rates
                        base_firing = 20.0  # Hz
                        network_size_effect = np.log10(neuron_count) * 2.0
                        batch_size_effect = np.log10(batch_size + 1) * 1.0
                        seq_effect = np.log10(seq_len/100 + 1) * 0.5
                        
                        mean_rate = base_firing - network_size_effect + batch_size_effect + seq_effect
                        mean_rate = max(5.0, min(40.0, mean_rate))  # Constrain to realistic range
                        mean_rate *= np.random.normal(1.0, 0.08)    # Add some variability
                        
                        # Ensure values are realistic
                        step_time = max(0.05, step_time)
                        memory = max(0.5, memory)
                        
                        step_times.append(step_time)
                        memory_usages.append(memory)
                        firing_rates.append(mean_rate)
                    
                    # Calculate mean and standard error for this configuration
                    mean_step_time = np.mean(step_times)
                    sem_step_time = np.std(step_times) / np.sqrt(len(step_times))
                    mean_gpu_memory = np.mean(memory_usages)
                    sem_gpu_memory = np.std(memory_usages) / np.sqrt(len(memory_usages))
                    mean_firing_rate = np.mean(firing_rates)
                    sem_firing_rate = np.std(firing_rates) / np.sqrt(len(firing_rates))
                    
                    # Add to the dataset
                    raw_data.append({
                        'sim_name': sim_name,
                        'n_neurons': neuron_count,
                        'n_edges': n_edges,
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'mean_step_time': mean_step_time,
                        'sem_step_time': sem_step_time,
                        'mean_gpu_memory': mean_gpu_memory,
                        'sem_gpu_memory': sem_gpu_memory,
                        'mean_rate': mean_firing_rate,
                        'sem_rate': sem_firing_rate,
                        'mode': mode
                    })
    
    # Create DataFrame
    df = pd.DataFrame(raw_data)
    
    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Synthetic training statistics saved to: {output_csv}")
        print(f"Generated {len(df)} rows of data with {len(modes)} modes, {len(seq_lens)} sequence lengths, {len(batch_sizes)} batch sizes, and {len(neuron_counts)} neuron counts")
        print(f"Features include: {', '.join(df.columns)}")
    
    return df

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
        ax1.plot(subset['seq_len'], subset['mean_step_time'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax1.fill_between(subset['seq_len'], 
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
        ax2.plot(subset['seq_len'], subset['mean_memory'], 
                marker='o', markersize=5, label=f"{neurons:,} neurons", 
                color=palette[i])
        ax2.fill_between(subset['seq_len'], 
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
    
    # Check if we have multiple sequence lengths to calculate scaling factors
    seq_len_values = sorted(df['seq_len'].unique())
    if len(seq_len_values) > 1:
        # Calculate scaling factors for sequence length 
        scaling_data = []
        baseline_seq_len = min(df['seq_len'].unique())
        
        # Compute scaling factors (average across neuron counts)
        for neurons in sorted(df['neurons'].unique()):
            neuron_data = df[(df['batch_size'] == 1) & (df['neurons'] == neurons)]
            baseline_data = neuron_data[neuron_data['seq_len'] == baseline_seq_len]
            
            # Check if baseline data exists
            if not baseline_data.empty:
                baseline_time = baseline_data['step_time'].mean()
                
                for seq_len in sorted(df['seq_len'].unique()):
                    if seq_len != baseline_seq_len:
                        seq_len_data = neuron_data[neuron_data['seq_len'] == seq_len]
                        if not seq_len_data.empty:
                            current_time = seq_len_data['step_time'].mean()
                            scaling_factor = current_time / baseline_time
                            scaling_data.append({
                                'neurons': neurons,
                                'seq_len': seq_len,
                                'scaling_factor': scaling_factor
                            })
        
        # Create DataFrame and check if we have data
        scaling_df = pd.DataFrame(scaling_data)
        
        if not scaling_df.empty and 'seq_len' in scaling_df.columns:
            # Compute average scaling factors across neuron counts
            avg_scaling = scaling_df.groupby('seq_len')['scaling_factor'].mean().reset_index()
            sem_scaling = scaling_df.groupby('seq_len')['scaling_factor'].sem().reset_index()
            
            # Plot scaling factors as bar chart
            bars = ax3.bar(avg_scaling['seq_len'].astype(str), avg_scaling['scaling_factor'], 
                          yerr=sem_scaling['scaling_factor'], capsize=4,
                          color=palette[1], edgecolor='black', linewidth=1)
            
            # Annoate bars with values
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.1f}x',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        else:
            # Not enough data for scaling factor visualization
            ax3.text(0.5, 0.5, "Insufficient data\nfor scaling factors",
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        # Not enough sequence length values
        ax3.text(0.5, 0.5, "Multiple sequence lengths\nrequired for scaling factors",
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time Scaling Factor')
    ax3.set_title('B) Time Scaling', fontweight='normal', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel D: Memory scaling factors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Check if we have data to calculate batch memory scaling factors
    if len(seq_len_values) > 0 and len(df['batch_size'].unique()) > 1:
        try:
            # Calculate scaling factors for batch size memory impact
            batch_mem_scaling_data = []
            baseline_batch = min(df['batch_size'].unique())
            
            for neurons in sorted(df['neurons'].unique()):
                neuron_data = df[(df['seq_len'] == middle_seq_len) & (df['neurons'] == neurons)]
                if not neuron_data.empty and baseline_batch in neuron_data['batch_size'].values:
                    baseline_mem = neuron_data[neuron_data['batch_size'] == baseline_batch]['memory'].mean()
                    
                    for batch_size in sorted(df['batch_size'].unique()):
                        if batch_size != baseline_batch and batch_size in neuron_data['batch_size'].values:
                            current_mem = neuron_data[neuron_data['batch_size'] == batch_size]['memory'].mean()
                            scaling_factor = current_mem / baseline_mem
                            batch_mem_scaling_data.append({
                                'neurons': neurons,
                                'batch_size': batch_size,
                                'scaling_factor': scaling_factor
                            })
            
            # If we have scaling data, create DataFrame and generate visualization
            if batch_mem_scaling_data:
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
                
                # Annotate bars with values
                for bar in bars:
                    height = bar.get_height()
                    ax4.annotate(f'{height:.1f}x',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),  
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
            else:
                # Not enough data for batch scaling factors
                ax4.text(0.5, 0.5, "Insufficient data\nfor memory scaling factors",
                        ha='center', va='center', transform=ax4.transAxes)
        except Exception as e:
            # Handle any errors during calculation
            print(f"Warning: Could not calculate batch memory scaling factors: {e}")
            ax4.text(0.5, 0.5, "Error calculating\nmemory scaling factors",
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        # Not enough batch size values
        ax4.text(0.5, 0.5, "Multiple batch sizes\nrequired for scaling factors",
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Memory Scaling Factor')
    ax4.set_title('D) Memory Scaling', fontweight='normal', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
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
    
    # Check if we have multiple sequence lengths
    if len(seq_len_values) > 0:
        # Use the middle sequence length for better visualization
        try:
            middle_seq_len = sorted(df['seq_len'].unique())[1]  # e.g., 500
        except IndexError:
            # If there's only one sequence length, use that
            middle_seq_len = sorted(df['seq_len'].unique())[0]
            
        # Prepare the data
        batch_data = df[df['seq_len'] == middle_seq_len].groupby(['batch_size', 'neurons']).agg(
            mean_step_time=('step_time', 'mean'),
            sem_step_time=('step_time', 'sem')
        ).reset_index()
        
        # Create line plot with error bands
        for i, neurons in enumerate(sorted(df['neurons'].unique())):
            subset = batch_data[batch_data['neurons'] == neurons]
            if not subset.empty:
                ax1.plot(subset['batch_size'], subset['mean_step_time'], 
                        marker='o', markersize=5, label=f"{neurons:,} neurons (Seq Len {middle_seq_len})", 
                        color=palette[i])
                ax1.fill_between(subset['batch_size'], 
                                subset['mean_step_time'] - subset['sem_step_time'],
                                subset['mean_step_time'] + subset['sem_step_time'],
                                alpha=0.2, color=palette[i])
    else:
        ax1.text(0.5, 0.5, "No sequence length data available",
                ha='center', va='center', transform=ax1.transAxes)
    
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
    
    # Check if we have sequence length data
    if len(seq_len_values) > 0:
        try:
            middle_seq_len = sorted(df['seq_len'].unique())[1]  # e.g., 500
        except IndexError:
            # If there's only one sequence length, use that
            middle_seq_len = sorted(df['seq_len'].unique())[0]
            
        # Prepare data
        batch_mem_data = df[df['seq_len'] == middle_seq_len].groupby(['batch_size', 'neurons']).agg(
            mean_memory=('memory', 'mean'),
            sem_memory=('memory', 'sem')
        ).reset_index()
        
        # Create line plot with error bands
        for i, neurons in enumerate(sorted(df['neurons'].unique())):
            subset = batch_mem_data[batch_mem_data['neurons'] == neurons]
            if not subset.empty:
                ax2.plot(subset['batch_size'], subset['mean_memory'], 
                        marker='o', markersize=5, label=f"{neurons:,} neurons", 
                        color=palette[i])
                ax2.fill_between(subset['batch_size'], 
                                subset['mean_memory'] - subset['sem_memory'],
                                subset['mean_memory'] + subset['sem_memory'],
                                alpha=0.2, color=palette[i])
    else:
        ax2.text(0.5, 0.5, "No sequence length data available",
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('C) Impact of Batch Size on Memory Usage', fontweight='normal', loc='left')
    
    # Panel C: Bar chart showing time scaling factor
    ax3 = fig.add_subplot(gs[0, 1])
    
    # Check if we have data to calculate batch scaling factors
    if len(seq_len_values) > 0 and len(df['batch_size'].unique()) > 1:
        try:
            # Calculate scaling factors for batch size
            batch_scaling_data = []
            baseline_batch = min(df['batch_size'].unique())
            
            for neurons in sorted(df['neurons'].unique()):
                neuron_data = df[(df['seq_len'] == middle_seq_len) & (df['neurons'] == neurons)]
                if not neuron_data.empty and baseline_batch in neuron_data['batch_size'].values:
                    baseline_time = neuron_data[neuron_data['batch_size'] == baseline_batch]['step_time'].mean()
                    
                    for batch_size in sorted(df['batch_size'].unique()):
                        if batch_size != baseline_batch and batch_size in neuron_data['batch_size'].values:
                            current_time = neuron_data[neuron_data['batch_size'] == batch_size]['step_time'].mean()
                            scaling_factor = current_time / baseline_time
                            batch_scaling_data.append({
                                'neurons': neurons,
                                'batch_size': batch_size,
                                'scaling_factor': scaling_factor
                            })
            
            # If we have scaling data, create a DataFrame and generate visualization
            if batch_scaling_data:
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
                
                # Annotate bars with values
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(f'{height:.1f}x',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),  
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
            else:
                # Not enough data for batch scaling factors
                ax3.text(0.5, 0.5, "Insufficient data\nfor batch scaling factors",
                        ha='center', va='center', transform=ax3.transAxes)
        except Exception as e:
            # Handle any errors during calculation
            print(f"Warning: Could not calculate batch scaling factors: {e}")
            ax3.text(0.5, 0.5, "Error calculating\nbatch scaling factors",
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        # Not enough batch size values
        ax3.text(0.5, 0.5, "Multiple batch sizes\nrequired for scaling factors",
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time Scaling Factor')
    ax3.set_title('B) Time Scaling', fontweight='normal', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel D: Memory scaling factors
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Check if we have data to calculate batch memory scaling factors
    if len(seq_len_values) > 0 and len(df['batch_size'].unique()) > 1:
        try:
            # Calculate scaling factors for batch size memory impact
            batch_mem_scaling_data = []
            baseline_batch = min(df['batch_size'].unique())
            
            for neurons in sorted(df['neurons'].unique()):
                neuron_data = df[(df['seq_len'] == middle_seq_len) & (df['neurons'] == neurons)]
                if not neuron_data.empty and baseline_batch in neuron_data['batch_size'].values:
                    baseline_mem = neuron_data[neuron_data['batch_size'] == baseline_batch]['memory'].mean()
                    
                    for batch_size in sorted(df['batch_size'].unique()):
                        if batch_size != baseline_batch and batch_size in neuron_data['batch_size'].values:
                            current_mem = neuron_data[neuron_data['batch_size'] == batch_size]['memory'].mean()
                            scaling_factor = current_mem / baseline_mem
                            batch_mem_scaling_data.append({
                                'neurons': neurons,
                                'batch_size': batch_size,
                                'scaling_factor': scaling_factor
                            })
            
            # If we have scaling data, create DataFrame and generate visualization
            if batch_mem_scaling_data:
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
                
                # Annotate bars with values
                for bar in bars:
                    height = bar.get_height()
                    ax4.annotate(f'{height:.1f}x',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),  
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
            else:
                # Not enough data for batch scaling factors
                ax4.text(0.5, 0.5, "Insufficient data\nfor memory scaling factors",
                        ha='center', va='center', transform=ax4.transAxes)
        except Exception as e:
            # Handle any errors during calculation
            print(f"Warning: Could not calculate batch memory scaling factors: {e}")
            ax4.text(0.5, 0.5, "Error calculating\nmemory scaling factors",
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        # Not enough batch size values
        ax4.text(0.5, 0.5, "Multiple batch sizes\nrequired for scaling factors",
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Memory Scaling Factor')
    ax4.set_title('D) Memory Scaling', fontweight='normal', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add legend for the whole figure
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
                marker='o', markersize=5, label=f"Seq Len {seq_len}", 
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
                marker='o', markersize=5, label=f"Seq Len {seq_len}", 
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

def create_comprehensive_figure(df, output_dir='figures/performance_summary'):
    """
    Create a comprehensive figure that shows the relationship between model parameters
    and performance metrics in a clear, publication-ready format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the performance data
    output_dir : str
        Directory to save the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate data by mode if present
    train_df = df[df['mode'] == 'train'] if 'mode' in df.columns else df
    test_df = df[df['mode'] == 'test'] if 'mode' in df.columns else pd.DataFrame()
    
    # Create a comprehensive figure for training data
    if not train_df.empty:
        _create_mode_specific_figure(train_df, output_dir, mode='train')
        
    # Create a comprehensive figure for testing data if available
    if not test_df.empty:
        _create_mode_specific_figure(test_df, output_dir, mode='test')
    
    # Create a comparison figure between train and test if both are available
    if not train_df.empty and not test_df.empty:
        _create_comparison_figure(train_df, test_df, output_dir)
        
def _create_mode_specific_figure(df, output_dir, mode='train'):
    """Create a comprehensive figure for a specific mode (train or test)"""
    # Set up color palette
    palette = sns.color_palette("viridis", max(len(df['neurons'].unique()), 5))
    
    # Create a figure with 6 panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # Panel 1: Step Time vs Neurons
    ax1 = fig.add_subplot(gs[0, :2])
    for i, seq_len in enumerate(sorted(df['seq_len'].unique())):
        if i >= len(df['seq_len'].unique()) - 3:  # Only plot top 3 sequence lengths
            subset = df[(df['batch_size'] == 1) & (df['seq_len'] == seq_len)]
            ax1.plot(subset['neurons'], subset['step_time'], 
                     marker='o', label=f"Seq Len {seq_len}", color=palette[i])
    
    ax1.set_xlabel('Number of Neurons')
    ax1.set_ylabel('Step Time (s)')
    ax1.set_title(f'A) Neuron Count vs Step Time ({mode.title()} Mode)', fontweight='normal', loc='left')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(title='Sequence Length')
    
    # Panel 2: Memory vs Neurons
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, seq_len in enumerate(sorted(df['seq_len'].unique())):
        if i >= len(df['seq_len'].unique()) - 3:
            subset = df[(df['batch_size'] == 1) & (df['seq_len'] == seq_len)]
            ax2.plot(subset['neurons'], subset['memory'], 
                     marker='o', label=f"Seq Len {seq_len}", color=palette[i])
    
    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.set_title(f'B) Neuron Count vs Memory ({mode.title()} Mode)', fontweight='normal', loc='left')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(title='Sequence Length')
    
    # Panel 3: Step Time vs Batch Size
    ax3 = fig.add_subplot(gs[1, :2])
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        if i % 2 == 0 and i < 6:  # Only show a subset of neuron counts
            subset = df[(df['neurons'] == neurons) & (df['seq_len'] == sorted(df['seq_len'].unique())[0])]
            ax3.plot(subset['batch_size'], subset['step_time'], 
                     marker='o', label=f"{neurons} neurons", color=palette[i])
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Step Time (s)')
    ax3.set_title(f'C) Batch Size vs Step Time ({mode.title()} Mode)', fontweight='normal', loc='left')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.legend(title='Neuron Count')
    
    # Panel 4: Memory vs Batch Size
    ax4 = fig.add_subplot(gs[1, 2:])
    for i, neurons in enumerate(sorted(df['neurons'].unique())):
        if i % 2 == 0 and i < 6:
            subset = df[(df['neurons'] == neurons) & (df['seq_len'] == sorted(df['seq_len'].unique())[0])]
            ax4.plot(subset['batch_size'], subset['memory'], 
                     marker='o', label=f"{neurons} neurons", color=palette[i])
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('GPU Memory (GB)')
    ax4.set_title(f'D) Batch Size vs Memory ({mode.title()} Mode)', fontweight='normal', loc='left')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.legend(title='Neuron Count')
    
    # Panel 5: Network Edges vs Step Time
    ax5 = fig.add_subplot(gs[2, :2])
    sizes = np.log10(df['neurons']) * 20
    scatter = ax5.scatter(df['n_edges'], df['step_time'], c=df['batch_size'], 
                          s=sizes, alpha=0.7, cmap='viridis')
    
    # Add legend
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.7)
    batch_legend = plt.legend(handles, [f'Batch {int(float(label))}' for label in labels], 
                             loc="upper left", title="Batch Size")
    ax5.add_artist(batch_legend)
    
    # Add regression line
    x = df['n_edges']
    y = df['step_time']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax5.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, 
             label=f"Linear fit (y = {z[0]:.2e}x + {z[1]:.2f})")
    
    ax5.set_xlabel('Number of Network Edges')
    ax5.set_ylabel('Step Time (s)')
    ax5.set_title(f'E) Network Edges vs Step Time ({mode.title()} Mode)', fontweight='normal', loc='left')
    ax5.grid(alpha=0.3, linestyle='--')
    ax5.legend(title='Fit')
    
    # Panel 6: Mean Rate Analysis
    ax6 = fig.add_subplot(gs[2, 2:])
    if 'mean_rate' in df.columns:
        sizes = np.log10(df['neurons']) * 20
        scatter = ax6.scatter(df['mean_rate'], df['step_time'], c=df['batch_size'], 
                              s=sizes, alpha=0.7, cmap='plasma')
        
        # Add legend
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.7)
        batch_legend = plt.legend(handles, [f'Batch {int(float(label))}' for label in labels], 
                                 loc="upper left", title="Batch Size")
        ax6.add_artist(batch_legend)
        
        # Add regression line if there's variation in mean_rate
        if df['mean_rate'].std() > 0:
            x = df['mean_rate']
            y = df['step_time']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax6.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, 
                    label=f"Linear fit (y = {z[0]:.2e}x + {z[1]:.2f})")
        
        ax6.set_xlabel('Mean Firing Rate (Hz)')
        ax6.set_ylabel('Step Time (s)')
        ax6.set_title(f'F) Firing Rate vs Step Time ({mode.title()} Mode)', 
                     fontweight='normal', loc='left')
        ax6.grid(alpha=0.3, linestyle='--')
        ax6.legend(title='Fit')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comprehensive_performance_{mode}.png'), dpi=300)
    # plt.savefig(os.path.join(output_dir, f'comprehensive_performance_{mode}.pdf'))
    plt.close()

def _create_comparison_figure(train_df, test_df, output_dir):
    """Create a figure comparing train and test mode performance"""
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Compare Step Time between train and test
    ax = axs[0, 0]
    
    # Get common neuron values in both datasets
    common_neurons = sorted(set(train_df['neurons'].unique()) & set(test_df['neurons'].unique()))
    
    # For each neuron count, compare train vs test performance
    train_times = []
    test_times = []
    labels = []
    
    for neuron in common_neurons:
        train_time = train_df[train_df['neurons'] == neuron]['step_time'].mean()
        test_time = test_df[test_df['neurons'] == neuron]['step_time'].mean()
        
        train_times.append(train_time)
        test_times.append(test_time)
        labels.append(f"{neuron:,}")
    
    # Plot comparison
    x = np.arange(len(common_neurons))
    width = 0.35
    
    ax.bar(x - width/2, train_times, width, label='Training')
    ax.bar(x + width/2, test_times, width, label='Testing')
    
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('Step Time (s)')
    ax.set_title('A) Train vs Test Mode: Step Time Comparison', fontweight='normal', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    ax.legend()
    
    # Panel 2: Compare Memory Usage between train and test
    ax = axs[0, 1]
    
    train_memory = []
    test_memory = []
    
    for neuron in common_neurons:
        train_mem = train_df[train_df['neurons'] == neuron]['memory'].mean()
        test_mem = test_df[test_df['neurons'] == neuron]['memory'].mean()
        
        train_memory.append(train_mem)
        test_memory.append(test_mem)
    
    # Plot comparison
    ax.bar(x - width/2, train_memory, width, label='Training')
    ax.bar(x + width/2, test_memory, width, label='Testing')
    
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('B) Train vs Test Mode: Memory Usage Comparison', fontweight='normal', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    ax.legend()
    
    # Panel 3: Compare Network Edges impact
    ax = axs[1, 0]
    
    # Create scatter plot with regression lines
    train_subset = train_df.sample(min(50, len(train_df))) if len(train_df) > 50 else train_df
    test_subset = test_df.sample(min(50, len(test_df))) if len(test_df) > 50 else test_df
    
    ax.scatter(train_subset['n_edges'], train_subset['step_time'], 
               alpha=0.5, label='Training', color='blue')
    ax.scatter(test_subset['n_edges'], test_subset['step_time'], 
               alpha=0.5, label='Testing', color='orange')
    
    # Add regression lines
    for data, color, label in [(train_df, 'blue', 'Training Fit'), 
                              (test_df, 'orange', 'Testing Fit')]:
        x = data['n_edges']
        y = data['step_time']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x), p(sorted(x)), '--', color=color, alpha=0.8, 
                label=f"{label} (y = {z[0]:.2e}x + {z[1]:.2f})")
    
    ax.set_xlabel('Number of Network Edges')
    ax.set_ylabel('Step Time (s)')
    ax.set_title('C) Network Edges Impact: Train vs Test', fontweight='normal', loc='left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    
    # Panel 4: Compare Firing Rate if available
    ax = axs[1, 1]
    
    if 'mean_rate' in train_df.columns and 'mean_rate' in test_df.columns:
        # Get mean firing rates for each mode and neuron count
        train_rates = []
        test_rates = []
        
        for neuron in common_neurons:
            train_rate = train_df[train_df['neurons'] == neuron]['mean_rate'].mean()
            test_rate = test_df[test_df['neurons'] == neuron]['mean_rate'].mean()
            
            train_rates.append(train_rate)
            test_rates.append(test_rate)
        
        # Plot comparison
        ax.bar(x - width/2, train_rates, width, label='Training')
        ax.bar(x + width/2, test_rates, width, label='Testing')
        
        ax.set_xlabel('Number of Neurons')
        ax.set_ylabel('Mean Firing Rate (Hz)')
        ax.set_title('D) Train vs Test Mode: Firing Rate Comparison', fontweight='normal', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(alpha=0.3, linestyle='--', axis='y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_test_comparison.png'), dpi=300)
    # plt.savefig(os.path.join(output_dir, 'train_test_comparison.pdf'))
    plt.close()

def plot_test_performance_figures(df, output_dir):
    """
    Create simplified performance figures for test data, focusing only on neuron count
    and network edges impact on performance metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the test performance data
    output_dir : str
        Directory to save the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a cohesive color palette
    palette = sns.color_palette("viridis", 4)
    
    # =========================================================================
    # FIGURE 1: Neuron Count vs Performance
    # =========================================================================
    
    plt.figure(figsize=(12, 5))
    
    # Panel 1: Step Time vs Neurons
    plt.subplot(1, 2, 1)
    
    # Group by neuron count
    neuron_data = df.groupby(['neurons']).agg(
        mean_step_time=('step_time', 'mean'),
        sem_step_time=('step_time', 'sem')
    ).reset_index()
    
    # Create bar plot with error bars
    bars = plt.bar(
        neuron_data['neurons'].astype(str), 
        neuron_data['mean_step_time'],
        yerr=neuron_data['sem_step_time'],
        capsize=4,
        color=palette[0],
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.2f}s',
            ha='center', 
            va='bottom',
            fontsize=9
        )
        
    plt.title('A) Neuron Count vs Step Time (Test Mode)', fontweight='normal', loc='left')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (s)')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    # Panel 2: Memory vs Neurons
    plt.subplot(1, 2, 2)
    
    # Group by neuron count for memory data
    memory_data = df.groupby(['neurons']).agg(
        mean_memory=('memory', 'mean'),
        sem_memory=('memory', 'sem')
    ).reset_index()
    
    # Create bar plot with error bars
    bars = plt.bar(
        memory_data['neurons'].astype(str), 
        memory_data['mean_memory'],
        yerr=memory_data['sem_memory'],
        capsize=4,
        color=palette[1],
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.05,
            f'{height:.1f}GB',
            ha='center', 
            va='bottom',
            fontsize=9
        )
        
    plt.title('B) Neuron Count vs Memory Usage (Test Mode)', fontweight='normal', loc='left')
    plt.xlabel('Number of Neurons')
    plt.ylabel('GPU Memory (GB)')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neuron_count_impact.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Batch Size Impact (if multiple batch sizes exist)
    # =========================================================================
    
    if len(df['batch_size'].unique()) > 1:
        plt.figure(figsize=(12, 5))
        
        # Panel 1: Step Time vs Batch Size for each neuron count
        plt.subplot(1, 2, 1)
        
        for i, neurons in enumerate(sorted(df['neurons'].unique())):
            subset = df[df['neurons'] == neurons]
            if not subset.empty:
                plt.plot(
                    subset['batch_size'], 
                    subset['step_time'], 
                    marker='o', 
                    label=f"{neurons:,} neurons",
                    color=palette[i % len(palette)]
                )
                
        plt.title('A) Batch Size vs Step Time (Test Mode)', fontweight='normal', loc='left')
        plt.xlabel('Batch Size')
        plt.ylabel('Step Time (s)')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(title='Neuron Count')
        
        # Panel 2: Memory vs Batch Size for each neuron count
        plt.subplot(1, 2, 2)
        
        for i, neurons in enumerate(sorted(df['neurons'].unique())):
            subset = df[df['neurons'] == neurons]
            if not subset.empty:
                plt.plot(
                    subset['batch_size'], 
                    subset['memory'], 
                    marker='o', 
                    label=f"{neurons:,} neurons",
                    color=palette[i % len(palette)]
                )
                
        plt.title('B) Batch Size vs Memory Usage (Test Mode)', fontweight='normal', loc='left')
        plt.xlabel('Batch Size')
        plt.ylabel('GPU Memory (GB)')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(title='Neuron Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_size_impact.png'), dpi=300)
        plt.close()
    
    # =========================================================================
    # FIGURE 3: Per-neuron metrics (efficiency analysis)
    # =========================================================================
    
    plt.figure(figsize=(12, 5))
    
    # Calculate per-neuron time and memory
    df['time_per_neuron'] = df['step_time'] / df['neurons']
    df['memory_per_neuron'] = df['memory'] / df['neurons']
    
    # Panel 1: Time per neuron
    plt.subplot(1, 2, 1)
    
    # Group by neuron count
    per_neuron_data = df.groupby(['neurons']).agg(
        mean_time_per_neuron=('time_per_neuron', 'mean'),
        sem_time_per_neuron=('time_per_neuron', 'sem')
    ).reset_index()
    
    # Create bar plot with error bars
    plt.bar(
        per_neuron_data['neurons'].astype(str), 
        per_neuron_data['mean_time_per_neuron'] * 1000,  # Convert to ms
        yerr=per_neuron_data['sem_time_per_neuron'] * 1000,
        capsize=4,
        color=palette[2],
        edgecolor='black',
        linewidth=1
    )
    
    plt.title('A) Computation Time per Neuron', fontweight='normal', loc='left')
    plt.xlabel('Network Size (neurons)')
    plt.ylabel('Time per Neuron (ms)')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    # Panel 2: Memory per neuron
    plt.subplot(1, 2, 2)
    
    # Group by neuron count
    per_neuron_mem_data = df.groupby(['neurons']).agg(
        mean_memory_per_neuron=('memory_per_neuron', 'mean'),
        sem_memory_per_neuron=('memory_per_neuron', 'sem')
    ).reset_index()
    
    # Create bar plot with error bars
    plt.bar(
        per_neuron_mem_data['neurons'].astype(str), 
        per_neuron_mem_data['mean_memory_per_neuron'] * 1000,  # Convert to MB
        yerr=per_neuron_mem_data['sem_memory_per_neuron'] * 1000,
        capsize=4,
        color=palette[3],
        edgecolor='black',
        linewidth=1
    )
    
    plt.title('B) Memory Usage per Neuron', fontweight='normal', loc='left')
    plt.xlabel('Network Size (neurons)')
    plt.ylabel('Memory per Neuron (MB)')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_neuron_metrics.png'), dpi=300)
    plt.close()

def plot_network_edges_impact_simplified(df, output_dir):
    """
    Create simplified figures showing the impact of network edges (n_edges) on performance metrics
    specifically for test data, which typically has fewer parameters varying.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the test performance data
    output_dir : str
        Directory to save the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set color palette
    palette = sns.color_palette("viridis", 4)
    
    # =========================================================================
    # FIGURE 1: Network Edges vs Step Time
    # =========================================================================
    plt.figure(figsize=(10, 5))
    
    # Group data by neuron count to get statistics
    grouped_data = df.groupby(['neurons']).agg({
        'n_edges': 'mean',  # For each neuron count, there should be a consistent edge count
        'step_time': ['mean', 'sem'],
        'memory': ['mean', 'sem']
    })
    
    # Flatten the multi-index columns
    grouped_data.columns = ['_'.join(col).strip('_') for col in grouped_data.columns.values]
    grouped_data = grouped_data.reset_index()
    
    # Panel 1: Plot network edges vs step time relationship
    plt.subplot(1, 2, 1)
    plt.errorbar(
        grouped_data['n_edges_mean'], 
        grouped_data['step_time_mean'],
        yerr=grouped_data['step_time_sem'],
        fmt='o-', 
        capsize=5,
        color=palette[0],
        linewidth=2,
        markersize=8
    )
    
    # Add text labels for each point showing neuron count
    for i, row in grouped_data.iterrows():
        plt.text(
            row['n_edges_mean'] * 1.05, 
            row['step_time_mean'], 
            f"{int(row['neurons']):,}",
            fontsize=9,
            ha='left',
            va='center'
        )
        
    # Add regression line
    x = grouped_data['n_edges_mean']
    y = grouped_data['step_time_mean']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.7, 
             label=f"Linear fit (y = {z[0]:.2e}x + {z[1]:.2f})")
    
    plt.xlabel('Number of Network Edges')
    plt.ylabel('Step Time (s)')
    plt.title('A) Network Edges vs Step Time', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    # Panel 2: Plot network edges vs memory relationship
    plt.subplot(1, 2, 2)
    plt.errorbar(
        grouped_data['n_edges_mean'], 
        grouped_data['memory_mean'],
        yerr=grouped_data['memory_sem'],
        fmt='o-', 
        capsize=5,
        color=palette[1],
        linewidth=2,
        markersize=8
    )
    
    # Add text labels for each point showing neuron count
    for i, row in grouped_data.iterrows():
        plt.text(
            row['n_edges_mean'] * 1.05, 
            row['memory_mean'], 
            f"{int(row['neurons']):,}",
            fontsize=9,
            ha='left',
            va='center'
        )
        
    # Add regression line
    x = grouped_data['n_edges_mean']
    y = grouped_data['memory_mean']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.7, 
             label=f"Linear fit (y = {z[0]:.2e}x + {z[1]:.2f})")
    
    plt.xlabel('Number of Network Edges')
    plt.ylabel('GPU Memory (GB)')
    plt.title('B) Network Edges vs Memory Usage', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_edges_impact.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Efficiency Metrics
    # =========================================================================
    plt.figure(figsize=(10, 5))
    
    # Calculate efficiency metrics
    grouped_data['time_per_edge'] = grouped_data['step_time_mean'] / grouped_data['n_edges_mean'] * 1e9  # ns per edge
    grouped_data['memory_per_edge'] = grouped_data['memory_mean'] / grouped_data['n_edges_mean'] * 1e6   # bytes per edge
    
    # Panel 1: Computation time per edge
    plt.subplot(1, 2, 1)
    plt.bar(
        grouped_data['neurons'].astype(str),
        grouped_data['time_per_edge'],
        color=palette[2],
        edgecolor='black',
        linewidth=1
    )
    
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time per Edge (ns)')
    plt.title('A) Computational Efficiency per Edge', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    # Panel 2: Memory per edge
    plt.subplot(1, 2, 2)
    plt.bar(
        grouped_data['neurons'].astype(str),
        grouped_data['memory_per_edge'],
        color=palette[3],
        edgecolor='black',
        linewidth=1
    )
    
    plt.xlabel('Number of Neurons')
    plt.ylabel('Memory per Edge (bytes)')
    plt.title('B) Memory Efficiency per Edge', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_edges_efficiency.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Edge Density Impact
    # =========================================================================
    if len(df['neurons'].unique()) >= 3:  # Only create this figure if we have enough neuron counts
        plt.figure(figsize=(8, 6))
        
        # Calculate edge density (edges per neuron)
        grouped_data['edge_density'] = grouped_data['n_edges_mean'] / grouped_data['neurons']
        
        # Create scatter plot with 3 metrics
        scatter = plt.scatter(
            grouped_data['neurons'],
            grouped_data['edge_density'],
            s=grouped_data['step_time_mean'] * 80,  # Size represents step time
            c=grouped_data['memory_mean'],         # Color represents memory usage
            cmap='plasma',
            alpha=0.8,
            edgecolors='black'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Memory Usage (GB)')
        
        # Add annotations
        for i, row in grouped_data.iterrows():
            plt.annotate(
                f"{int(row['neurons']):,} neurons",
                (row['neurons'], row['edge_density']),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
                ha='left'
            )
            
        plt.xscale('log')
        plt.xlabel('Number of Neurons (log scale)')
        plt.ylabel('Edge Density (edges per neuron)')
        plt.title('Network Size vs Edge Density', fontweight='normal')
        plt.grid(alpha=0.3, linestyle='--')
        
        # Add a note about the bubble size
        plt.annotate(
            'Bubble size represents step time',
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3)
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'edge_density_impact.png'), dpi=300)
        plt.close()

def create_normalized_step_time_comparison(train_df, test_df, output_dir):
    """
    Create a specialized comparison between train and test data for sequence length 4000,
    with step time normalized by batch size to provide a fair comparison.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        DataFrame containing the training data
    test_df : pandas.DataFrame
        DataFrame containing the testing data
    output_dir : str
        Directory to save the generated figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Filter for sequence length 1000 in training data
    train_seq1000 = train_df[train_df['seq_len'] == 1000] if 'seq_len' in train_df.columns else pd.DataFrame()
    # Prepare test data (might not have seq_len column if it's constant)
    test_data = test_df.copy()
    # Normalize step times by batch size for fair comparison
    if not train_seq1000.empty:
        train_seq1000['normalized_step_time'] = train_seq1000['step_time'] / train_seq1000['batch_size']
    if not test_data.empty:
        test_data['normalized_step_time'] = test_data['step_time'] / test_data['batch_size']
    
    # Create figure for comparison
    if not train_seq1000.empty and not test_data.empty:
        plt.figure(figsize=(12, 10))
        # Panel 1: Normalized Step Time vs Neuron Count
        plt.subplot(2, 1, 1)
        # Get common neuron values
        common_neurons = sorted(set(train_seq1000['neurons'].unique()) & set(test_data['neurons'].unique()))
        # Calculate average normalized step times for each neuron count
        train_norm_times = []
        test_norm_times = []
        
        for neuron in common_neurons:
            # Filter data for this neuron count
            train_neurons = train_seq1000[train_seq1000['neurons'] == neuron]
            test_neurons = test_data[test_data['neurons'] == neuron]
            # Calculate mean normalized step time for each mode
            train_norm_time = train_neurons['normalized_step_time'].mean()
            test_norm_time = test_neurons['normalized_step_time'].mean()
            train_norm_times.append(train_norm_time)
            test_norm_times.append(test_norm_time)
        
        # Plot bar chart comparison
        x = np.arange(len(common_neurons))
        width = 0.35

        plt.bar(x - width/2, train_norm_times, width, label='Training (seq_len=1000)', color='royalblue')
        plt.bar(x + width/2, test_norm_times, width, label='Testing', color='darkorange')
        
        plt.xlabel('Number of Neurons')
        plt.ylabel('Normalized Step Time (s/batch)')
        plt.title('Normalized Step Time Comparison: Train vs Test', fontweight='normal')
        plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
        plt.grid(alpha=0.3, linestyle='--', axis='y')
        plt.legend()
        
        # Add text annotations with efficiency gain
        for i, (train_time, test_time) in enumerate(zip(train_norm_times, test_norm_times)):
            efficiency = (train_time - test_time) / train_time * 100
            plt.annotate(
                f"{efficiency:.1f}% faster",
                xy=(x[i], max(train_time, test_time) * 1.05),
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Panel 2: Line plot showing scaling with network size
        plt.subplot(2, 1, 2)
        
        # Plot normalized step time vs neuron count for both modes
        plt.plot(common_neurons, train_norm_times, 'o-', label='Training (seq_len=1000)', color='royalblue')
        plt.plot(common_neurons, test_norm_times, 'o-', label='Testing', color='darkorange')
        
        # Add efficiency ratio as a secondary plot
        efficiency_ratios = [test/train for test, train in zip(test_norm_times, train_norm_times)]
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(common_neurons, efficiency_ratios, 'r--', label='Test/Train ratio')
        ax2.set_ylabel('Testing/Training Ratio', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(False)
        
        # Add a horizontal line at ratio=1.0
        ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
        
        plt.xscale('log')
        plt.xlabel('Number of Neurons (log scale)')
        plt.ylabel('Normalized Step Time (s/batch)')
        plt.title('Scaling of Normalized Step Time with Network Size', fontweight='normal')
        plt.grid(alpha=0.3, linestyle='--')
        
        # Combine legends from both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_step_time_comparison_seq4000.png'), dpi=300)
        plt.close()
        
        # Create a third visualization: efficiency metrics across network sizes
        plt.figure(figsize=(10, 6))
        
        # Calculate efficiency metrics (time per neuron) for both modes
        train_time_per_neuron = [t/n for t, n in zip(train_norm_times, common_neurons)]
        test_time_per_neuron = [t/n for t, n in zip(test_norm_times, common_neurons)]
        
        # Create a grouped bar chart
        plt.subplot(1, 1, 1)
        bar_width = 0.35
        x = np.arange(len(common_neurons))
        
        plt.bar(x - bar_width/2, 
                [t * 1000 for t in train_time_per_neuron], # convert to ms
                bar_width, 
                label='Training', 
                color='royalblue')
        
        plt.bar(x + bar_width/2, 
                [t * 1000 for t in test_time_per_neuron], # convert to ms
                bar_width, 
                label='Testing', 
                color='darkorange')
        
        plt.xlabel('Number of Neurons')
        plt.ylabel('Time per Neuron (ms/batch/neuron)')
        plt.title('Computational Efficiency per Neuron: Train vs Test', fontweight='normal')
        plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
        plt.grid(alpha=0.3, linestyle='--', axis='y')
        plt.legend()
        
        # Add efficiency improvement annotations
        for i, (train_tpn, test_tpn) in enumerate(zip(train_time_per_neuron, test_time_per_neuron)):
            efficiency = (train_tpn - test_tpn) / train_tpn * 100
            plt.annotate(
                f"{efficiency:.1f}% better",
                xy=(x[i], max(train_tpn, test_tpn) * 1000 * 1.05), # convert to ms
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'neuron_efficiency_comparison_seq4000.png'), dpi=300)
        plt.close()

def create_train_test_comparison(train_df, test_df, output_dir):
    """
    Create comparison figures between training and testing modes.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        DataFrame containing the training data
    test_df : pandas.DataFrame
        DataFrame containing the testing data
    output_dir : str
        Directory to save the generated figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get common neuron counts in both datasets
    common_neurons = sorted(set(train_df['neurons'].unique()) & set(test_df['neurons'].unique()))
    
    # =========================================================================
    # FIGURE 1: Step Time Comparison
    # =========================================================================
    plt.figure(figsize=(12, 6))
    
    # Calculate average step times for each neuron count
    train_times = []
    test_times = []
    train_times_err = []
    test_times_err = []
    
    for neuron in common_neurons:
        # Get data for this neuron count
        train_subset = train_df[train_df['neurons'] == neuron]
        test_subset = test_df[test_df['neurons'] == neuron]
        
        # Calculate mean and standard error
        train_times.append(train_subset['step_time'].mean())
        test_times.append(test_subset['step_time'].mean())
        train_times_err.append(train_subset['step_time'].sem())
        test_times_err.append(test_subset['step_time'].sem())
    
    # Create bar chart
    x = np.arange(len(common_neurons))
    width = 0.35
    
    plt.bar(x - width/2, train_times, width, yerr=train_times_err, 
           label='Training', color='royalblue', capsize=5)
    plt.bar(x + width/2, test_times, width, yerr=test_times_err,
           label='Testing', color='darkorange', capsize=5)
    
    # Add percentage improvement annotations
    for i, (train, test) in enumerate(zip(train_times, test_times)):
        if train > test:
            improvement = (train - test) / train * 100
            plt.text(x[i], max(train, test) * 1.05, 
                    f"{improvement:.1f}% faster", 
                    ha='center', fontsize=9)
    
    plt.xlabel('Number of Neurons')
    plt.ylabel('Step Time (s)')
    plt.title('Step Time Comparison: Training vs Testing', fontweight='normal')
    plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_time_comparison.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Memory Usage Comparison
    # =========================================================================
    plt.figure(figsize=(12, 6))
    
    # Calculate average memory usage for each neuron count
    train_mem = []
    test_mem = []
    train_mem_err = []
    test_mem_err = []
    
    for neuron in common_neurons:
        # Get data for this neuron count
        train_subset = train_df[train_df['neurons'] == neuron]
        test_subset = test_df[test_df['neurons'] == neuron]
        
        # Calculate mean and standard error
        train_mem.append(train_subset['memory'].mean())
        test_mem.append(test_subset['memory'].mean())
        train_mem_err.append(train_subset['memory'].sem())
        test_mem_err.append(test_subset['memory'].sem())
    
    # Create bar chart
    plt.bar(x - width/2, train_mem, width, yerr=train_mem_err,
           label='Training', color='royalblue', capsize=5)
    plt.bar(x + width/2, test_mem, width, yerr=test_mem_err,
           label='Testing', color='darkorange', capsize=5)
    
    # Add percentage improvement annotations
    for i, (train, test) in enumerate(zip(train_mem, test_mem)):
        if train > test:
            improvement = (train - test) / train * 100
            plt.text(x[i], max(train, test) * 1.05, 
                    f"{improvement:.1f}% less", 
                    ha='center', fontsize=9)
    
    plt.xlabel('Number of Neurons')
    plt.ylabel('GPU Memory (GB)')
    plt.title('Memory Usage Comparison: Training vs Testing', fontweight='normal')
    plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Performance across Network Sizes
    # =========================================================================
    plt.figure(figsize=(12, 10))
    
    # Panel 1: Step time vs. Network Size (log scale)
    plt.subplot(2, 1, 1)
    
    # Plot with error bars
    plt.errorbar(common_neurons, train_times, yerr=train_times_err, 
                fmt='o-', color='royalblue', label='Training', capsize=5)
    plt.errorbar(common_neurons, test_times, yerr=test_times_err,
                fmt='o-', color='darkorange', label='Testing', capsize=5)
    
    # Add trend lines
    train_z = np.polyfit(np.log10(common_neurons), np.log10(train_times), 1)
    test_z = np.polyfit(np.log10(common_neurons), np.log10(test_times), 1)
    
    train_p = np.poly1d(train_z)
    test_p = np.poly1d(test_z)
    
    x_smooth = np.logspace(np.log10(min(common_neurons)), np.log10(max(common_neurons)), 100)
    plt.plot(x_smooth, 10**train_p(np.log10(x_smooth)), '--', color='royalblue', 
            label=f'Training: ~N^{train_z[0]:.2f}')
    plt.plot(x_smooth, 10**test_p(np.log10(x_smooth)), '--', color='darkorange',
            label=f'Testing: ~N^{test_z[0]:.2f}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Neurons (log scale)')
    plt.ylabel('Step Time (s) (log scale)')
    plt.title('A) Computational Time Scaling with Network Size', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    # Panel 2: Memory usage vs. Network Size (log scale)
    plt.subplot(2, 1, 2)
    
    # Plot with error bars
    plt.errorbar(common_neurons, train_mem, yerr=train_mem_err, 
                fmt='o-', color='royalblue', label='Training', capsize=5)
    plt.errorbar(common_neurons, test_mem, yerr=test_mem_err,
                fmt='o-', color='darkorange', label='Testing', capsize=5)
    
    # Add trend lines
    train_mem_z = np.polyfit(np.log10(common_neurons), np.log10(train_mem), 1)
    test_mem_z = np.polyfit(np.log10(common_neurons), np.log10(test_mem), 1)
    
    train_mem_p = np.poly1d(train_mem_z)
    test_mem_p = np.poly1d(test_mem_z)
    
    plt.plot(x_smooth, 10**train_mem_p(np.log10(x_smooth)), '--', color='royalblue', 
            label=f'Training: ~N^{train_mem_z[0]:.2f}')
    plt.plot(x_smooth, 10**test_mem_p(np.log10(x_smooth)), '--', color='darkorange',
            label=f'Testing: ~N^{test_mem_z[0]:.2f}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Neurons (log scale)')
    plt.ylabel('GPU Memory (GB) (log scale)')
    plt.title('B) Memory Usage Scaling with Network Size', fontweight='normal', loc='left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_scaling.png'), dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Efficiency Metrics
    # =========================================================================
    if 'n_edges' in train_df.columns and 'n_edges' in test_df.columns:
        plt.figure(figsize=(12, 10))
        
        # Panel 1: Time per neuron
        plt.subplot(2, 1, 1)
        
        # Calculate time per neuron
        train_time_per_neuron = [t/n for t, n in zip(train_times, common_neurons)]
        test_time_per_neuron = [t/n for t, n in zip(test_times, common_neurons)]
        
        # Create bar chart
        plt.bar(x - width/2, [t*1000 for t in train_time_per_neuron], width, 
               label='Training', color='royalblue')
        plt.bar(x + width/2, [t*1000 for t in test_time_per_neuron], width,
               label='Testing', color='darkorange')
        
        # Add efficiency improvement annotations
        for i, (train, test) in enumerate(zip(train_time_per_neuron, test_time_per_neuron)):
            if train > test:
                improvement = (train - test) / train * 100
                plt.text(x[i], max(train, test) * 1000 * 1.05, 
                        f"{improvement:.1f}% better", 
                        ha='center', fontsize=9)
        
        plt.xlabel('Number of Neurons')
        plt.ylabel('Time per Neuron (ms)')
        plt.title('A) Computational Efficiency per Neuron', fontweight='normal', loc='left')
        plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
        plt.grid(alpha=0.3, linestyle='--', axis='y')
        plt.legend()
        
        # Panel 2: Memory per edge
        plt.subplot(2, 1, 2)
        
        # Get average edges for each neuron count
        train_edges = []
        test_edges = []
        
        for neuron in common_neurons:
            train_subset = train_df[train_df['neurons'] == neuron]
            test_subset = test_df[test_df['neurons'] == neuron]
            
            # Calculate mean edge count
            train_edges.append(train_subset['n_edges'].mean())
            test_edges.append(test_subset['n_edges'].mean())
        
        # Calculate memory per edge (in KB)
        train_mem_per_edge = [(m/e) * 1024 * 1024 for m, e in zip(train_mem, train_edges)]
        test_mem_per_edge = [(m/e) * 1024 * 1024 for m, e in zip(test_mem, test_edges)]
        
        # Create bar chart
        plt.bar(x - width/2, train_mem_per_edge, width, 
               label='Training', color='royalblue')
        plt.bar(x + width/2, test_mem_per_edge, width,
               label='Testing', color='darkorange')
        
        # Add efficiency improvement annotations
        for i, (train, test) in enumerate(zip(train_mem_per_edge, test_mem_per_edge)):
            if train > test:
                improvement = (train - test) / train * 100
                plt.text(x[i], max(train, test) * 1.05, 
                        f"{improvement:.1f}% better", 
                        ha='center', fontsize=9)
        
        plt.xlabel('Number of Neurons')
        plt.ylabel('Memory per Edge (KB)')
        plt.title('B) Memory Efficiency per Edge', fontweight='normal', loc='left')
        plt.xticks(x, [f"{n:,}" for n in common_neurons], rotation=45)
        plt.grid(alpha=0.3, linestyle='--', axis='y')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_metrics.png'), dpi=300)
        plt.close()
    

def main():
    """
    Main function to load real training data and create performance analysis figures.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate performance analysis figures from training data")
    parser.add_argument("--csv", default="/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/performance_figures/performance_statistics_thesis_data.csv", 
                       help="Path to CSV file containing training statistics")
    parser.add_argument("--output-dir", default="/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/performance_figures_new", help="Directory to save generated figures")
    args = parser.parse_args()
    
    # Create output directory
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load real training data
    print(f"Loading performance data from {args.csv}...")
    df = load_training_data(args.csv)
    
    if df.empty:
        print("Error: No data loaded. Please check the CSV file path.")
        return
        
    # Rename columns to match the expected format in the plot functions
    df = df.rename(columns={
        'n_neurons': 'neurons',
        'mean_step_time': 'step_time', 
        'sem_step_time': 'step_time_sem',
        'mean_gpu_memory': 'memory',
        'sem_gpu_memory': 'memory_sem'
    })

    print(df)
    
    # Split data by mode if available
    if 'mode' in df.columns:
        modes = df['mode'].unique()
        print(f"Found {len(modes)} operation modes: {', '.join(modes)}")
        
        # Separate train and test data
        train_df = df[df['mode'] == 'train']
        test_df = df[df['mode'] == 'test']
        
        # Generate train-specific figures (full set)
        if not train_df.empty:
            print("Creating training performance analysis figures...")
            output_dir_train = os.path.join(base_output_dir, "performance_train")
            plot_performance_figures(train_df, output_dir_train)
            print(f"  Saved training figures to {output_dir_train}")
                    
        # Generate test-specific figures (simplified set - only 4 key figures)
        if not test_df.empty:
            print("Creating testing performance analysis figures (simplified)...")
            output_dir_test = os.path.join(base_output_dir, "performance_test")
            plot_test_performance_figures(test_df, output_dir_test)
            print(f"  Saved testing figures to {output_dir_test}")
            
            # Plot network edges impact for testing (simplified)
            plot_network_edges_impact_simplified(test_df, os.path.join(base_output_dir, "network_edges_test"))
        
        # Generate train vs test comparison figures
        if not train_df.empty and not test_df.empty:
            print("Creating train vs test comparison figures...")
            train_test_dir = os.path.join(base_output_dir, "train_vs_test")
            os.makedirs(train_test_dir, exist_ok=True)
            
            # Create general comparison figures
            create_train_test_comparison(train_df, test_df, train_test_dir)
            
            # Create specific comparison for seq_len 4000 with normalized step time
            if 4000 in train_df['seq_len'].unique() and len(test_df) > 0:  # Test data might not have seq_len column
                create_normalized_step_time_comparison(train_df, test_df, train_test_dir)
            
            print(f"  Saved comparison figures to {train_test_dir}")
    else:
        print("No mode column found in the data. Processing all data together.")
        # Generate figures for all data combined
        print("Creating general performance analysis figures...")
        output_dir_all = os.path.join(base_output_dir, "performance_all")
        plot_performance_figures(df, output_dir_all)
        print(f"  Saved general figures to {output_dir_all}")
            
    print("\nPerformance analysis complete!")
    print(f"All figures have been saved to {base_output_dir}/")

if __name__ == "__main__":
    main()