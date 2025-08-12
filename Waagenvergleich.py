"""
Scale Weight Data Analysis Script

This script analyzes scale weight measurements from two CSV files containing TrueDyne sensor data.
It processes the "Scale Weight [g]" column from both datasets and performs comprehensive statistical
and visual analysis.

Features:
- Loads two CSV files with TrueDyne sensor measurements
- Extracts and compares scale weight data (m_1 and m_2)
- Removes outliers (values >50% away from average)
- Calculates rolling averages for data smoothing
- Computes statistical measures (mean, standard deviation)
- Centers data around zero by subtracting individual averages
- Creates comprehensive visualizations using Plotly including:
  * Time series plots of original and processed data
  * Rolling average comparisons
  * Zero-centered data visualization
  * Distribution histograms
  * Statistical summary tables

Input CSV Format Expected:
The CSV files should contain columns including:
- "Relative Time [s]": Time measurements
- "Scale Weight [g]": Weight measurements to be analyzed
- Other TrueDyne sensor columns (will be ignored)

Output:
- Interactive Plotly dashboard with 6 visualization panels
- Console statistics summary
- HTML file export of the analysis

Author: Generated for Physics Master's Thesis Analysis
Date: July 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_and_process_data(file1_path, file2_path, rolling_window=10):
    """
    Load CSV files and process Scale Weight data

    Parameters:
    file1_path (str): Path to first CSV file
    file2_path (str): Path to second CSV file
    rolling_window (int): Window size for rolling average

    Returns:
    dict: Processed data for both files
    """

    # Read CSV files with different encoding options to handle special characters
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    df1 = None
    df2 = None

    # Try different encodings for file 1
    for encoding in encodings_to_try:
        try:
            df1 = pd.read_csv(file1_path, encoding=encoding)
            print(f"Successfully loaded file 1 with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue

    if df1 is None:
        raise ValueError("Could not read file 1 with any supported encoding")

    # Try different encodings for file 2
    for encoding in encodings_to_try:
        try:
            df2 = pd.read_csv(file2_path, encoding=encoding)
            print(f"Successfully loaded file 2 with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue

    if df2 is None:
        raise ValueError("Could not read file 2 with any supported encoding")

    # Extract relevant columns
    time1 = df1['Relative Time [s]']
    time2 = df2['Relative Time [s]']
    m_1 = df1['Scale Weight [g]']
    m_2 = df2['Scale Weight [g]']

    # Remove outliers (values more than 50% away from average)
    def remove_outliers(data):
        avg = np.mean(data)
        threshold = 0.5 * avg  # 50% of average
        mask = np.abs(data - avg) <= threshold
        return data[mask], mask

    m_1_clean, mask1 = remove_outliers(m_1)
    m_2_clean, mask2 = remove_outliers(m_2)
    time1_clean = time1[mask1]
    time2_clean = time2[mask2]

    # Calculate rolling averages
    m_1_rolling = pd.Series(m_1_clean).rolling(window=rolling_window, center=True).mean()
    m_2_rolling = pd.Series(m_2_clean).rolling(window=rolling_window, center=True).mean()

    # Calculate standard deviations
    std_1 = np.std(m_1_clean)
    std_2 = np.std(m_2_clean)

    # Calculate averages for centering
    avg_1 = np.mean(m_1_clean)
    avg_2 = np.mean(m_2_clean)

    # Center data around zero
    m_1_centered = m_1_clean - avg_1
    m_2_centered = m_2_clean - avg_2

    return {
        'time1': time1_clean,
        'time2': time2_clean,
        'm_1_original': m_1_clean,
        'm_2_original': m_2_clean,
        'm_1_rolling': m_1_rolling,
        'm_2_rolling': m_2_rolling,
        'm_1_centered': m_1_centered,
        'm_2_centered': m_2_centered,
        'std_1': std_1,
        'std_2': std_2,
        'avg_1': avg_1,
        'avg_2': avg_2,
        'outliers_removed_1': len(m_1) - len(m_1_clean),
        'outliers_removed_2': len(m_2) - len(m_2_clean)
    }


def create_plots(data, downsample_factor=10):
    """
    Create comprehensive plots for the scale weight analysis

    Parameters:
    data (dict): Processed data dictionary
    downsample_factor (int): Factor to reduce data points for plotting (default: 10)
    """

    # Downsample data for plotting to improve performance
    def downsample_data(time_data, weight_data, factor):
        """Downsample data by taking every nth point"""
        indices = np.arange(0, len(time_data), factor)
        return time_data.iloc[indices] if hasattr(time_data, 'iloc') else time_data[indices], \
            weight_data.iloc[indices] if hasattr(weight_data, 'iloc') else weight_data[indices]

    print(f"Downsampling data by factor of {downsample_factor} for plotting performance...")

    # Downsample all datasets
    time1_plot, m_1_orig_plot = downsample_data(data['time1'], data['m_1_original'], downsample_factor)
    time2_plot, m_2_orig_plot = downsample_data(data['time2'], data['m_2_original'], downsample_factor)
    _, m_1_roll_plot = downsample_data(data['time1'], data['m_1_rolling'], downsample_factor)
    _, m_2_roll_plot = downsample_data(data['time2'], data['m_2_rolling'], downsample_factor)
    _, m_1_cent_plot = downsample_data(data['time1'], data['m_1_centered'], downsample_factor)
    _, m_2_cent_plot = downsample_data(data['time2'], data['m_2_centered'], downsample_factor)

    print(
        f"Plotting {len(time1_plot)} and {len(time2_plot)} points (downsampled from {len(data['time1'])} and {len(data['time2'])})")

    # Create subplots - focused on centered data analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Centered Data (Zero Mean) vs Time', 'Centered Data Distribution',
            'Centered Data Comparison', 'Statistics Summary'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )

    # Plot 1: Centered data (downsampled)
    fig.add_trace(
        go.Scatter(x=time1_plot, y=m_1_cent_plot,
                   mode='lines', name='m_1 (centered)', line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time2_plot, y=m_2_cent_plot,
                   mode='lines', name='m_2 (centered)', line=dict(color='red', width=1.5)),
        row=1, col=1
    )
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)

    # Plot 2: Histograms for centered data distribution (sample for performance)
    sample_size = min(10000, len(data['m_1_centered']), len(data['m_2_centered']))
    m_1_cent_sample = np.random.choice(data['m_1_centered'], size=min(sample_size, len(data['m_1_centered'])),
                                       replace=False)
    m_2_cent_sample = np.random.choice(data['m_2_centered'], size=min(sample_size, len(data['m_2_centered'])),
                                       replace=False)

    fig.add_trace(
        go.Histogram(x=m_1_cent_sample, name='m_1 centered distribution',
                     opacity=0.7, nbinsx=50, marker_color='blue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=m_2_cent_sample, name='m_2 centered distribution',
                     opacity=0.7, nbinsx=50, marker_color='red'),
        row=1, col=2
    )

    # Plot 3: Overlay comparison of centered data (downsampled)
    fig.add_trace(
        go.Scatter(x=time1_plot, y=m_1_cent_plot,
                   mode='lines', name='m_1 (centered)', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time2_plot, y=m_2_cent_plot,
                   mode='lines', name='m_2 (centered)', line=dict(color='red', width=2)),
        row=2, col=1
    )
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

    # Plot 4: Statistics table
    stats_table = [
        ['Metric', 'm_1', 'm_2'],
        ['Original Average (g)', f'{data["avg_1"]:.4f}', f'{data["avg_2"]:.4f}'],
        ['Std Dev (g)', f'{data["std_1"]:.4f}', f'{data["std_2"]:.4f}'],
        ['Centered Std Dev (g)', f'{np.std(data["m_1_centered"]):.4f}', f'{np.std(data["m_2_centered"]):.4f}'],
        ['Outliers Removed', str(data['outliers_removed_1']), str(data['outliers_removed_2'])],
        ['Data Points', str(len(data['m_1_original'])), str(len(data['m_2_original']))],
        ['Offset Removed (g)', f'{data["avg_1"]:.4f}', f'{data["avg_2"]:.4f}']
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=stats_table[0], fill_color='lightblue'),
            cells=dict(values=list(zip(*stats_table[1:])), fill_color='white')
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,  # Reduced height since fewer plots
        title_text="Scale Weight Analysis - Centered Data Focus",
        showlegend=True
    )

    # Update axis labels
    fig.update_xaxes(title_text="Relative Time [s]", row=1, col=1)
    fig.update_xaxes(title_text="Centered Weight [g]", row=1, col=2)
    fig.update_xaxes(title_text="Relative Time [s]", row=2, col=1)

    fig.update_yaxes(title_text="Centered Weight [g]", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Centered Weight [g]", row=2, col=1)

    return fig


def print_summary_statistics(data):
    """
    Print summary statistics
    """
    print("=" * 60)
    print("SCALE WEIGHT ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Dataset 1 (m_1):")
    print(f"  Average: {data['avg_1']:.4f} g")
    print(f"  Std Dev: {data['std_1']:.4f} g")
    print(f"  Data points: {len(data['m_1_original'])}")
    print(f"  Outliers removed: {data['outliers_removed_1']}")
    print()
    print(f"Dataset 2 (m_2):")
    print(f"  Average: {data['avg_2']:.4f} g")
    print(f"  Std Dev: {data['std_2']:.4f} g")
    print(f"  Data points: {len(data['m_2_original'])}")
    print(f"  Outliers removed: {data['outliers_removed_2']}")
    print()
    print(f"Difference in averages: {abs(data['avg_1'] - data['avg_2']):.4f} g")
    print(f"Ratio of std devs: {data['std_1'] / data['std_2']:.4f}")
    print("=" * 60)


# Main execution
if __name__ == "__main__":
    # File paths - modify these to match your file locations
    file1_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250730_Messreihe_19\measurements_20250729_142819\measurements_20250729_142819\DMA_Results_20250729_142759_truedyne_buffer.csv"
    file2_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250725_Messreihe_18\measurements_20250725_154909\measurements_20250725_154909\DMA_Results_20250725_154844_truedyne_buffer.csv"

    try:
        # Process the data
        print("Loading and processing data...")
        processed_data = load_and_process_data(file1_path, file2_path, rolling_window=10)

        # Print summary statistics
        print_summary_statistics(processed_data)

        # Create and show plots with downsampling for performance
        print("Creating plots...")
        fig = create_plots(processed_data, downsample_factor=20)  # Adjust factor as needed

        # Show plot with reduced rendering for better performance
        fig.update_layout(
            template="plotly_white",  # Simpler template
            font=dict(size=10),  # Smaller font
        )


        # Optional: Save the plot as HTML
        fig.write_html("scale_weight_analysis.html")
        print("Plot saved as 'scale_weight_analysis.html'")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        print(f"Please check the file paths and ensure the files exist.")
        print(f"File 1: {file1_path}")
        print(f"File 2: {file2_path}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("Please check your CSV files have the expected format and column names.")