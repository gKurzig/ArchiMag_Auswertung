import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import scipy.stats as stats


def plot_scale_data(csv_file_path, bin_size=100, use_webgl=True):
    """
    Plot Scale Weight vs Timestamp with highlighted non-dynamic points and 2-std area.
    Instead of downsampling, takes average of bin_size values with error bars.

    Args:
        csv_file_path (str): Path to the CSV file
        bin_size (int): Number of consecutive points to average (default: 100)
        use_webgl (bool): Use WebGL for faster rendering of large datasets
    """
    # Read the CSV file with different encodings to handle special characters
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, encoding='cp1252')

    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    print(f"Original data points: {len(df)}")

    # Create binned data by averaging every bin_size consecutive points
    def create_binned_data(data, bin_size):
        """Create binned data with mean and std for each bin"""
        n_bins = len(data) // bin_size
        if n_bins == 0:
            return data.copy()  # Return original if too few points

        # Trim data to fit complete bins
        trimmed_data = data.iloc[:n_bins * bin_size].copy()

        # Create bin labels
        trimmed_data['bin'] = np.repeat(range(n_bins), bin_size)

        # Group by bins and calculate statistics
        binned = trimmed_data.groupby('bin').agg({
            'Timestamp': 'mean',  # Average timestamp for each bin
            'Scale Weight [g]': ['mean', 'std'],
            'TrueDyne MGCE1 Density [kg/m³]': ['mean', 'std'],
            'Scale Status': lambda x: x.mode()[0] if not x.mode().empty else 'Dynamic'  # Most common status
        }).reset_index()

        # Flatten column names
        binned.columns = ['bin', 'Timestamp', 'Weight_mean', 'Weight_std', 'Density_mean', 'Density_std',
                          'Scale Status']

        # Fill NaN std values with 0 (for single-value bins)
        binned['Weight_std'] = binned['Weight_std'].fillna(0)
        binned['Density_std'] = binned['Density_std'].fillna(0)

        return binned

    # Separate dynamic and non-dynamic data
    non_dynamic_mask = df['Scale Status'] != 'Dynamic'
    dynamic_mask = df['Scale Status'] == 'Dynamic'

    # Process non-dynamic points (keep individual points)
    df_non_dynamic = df[non_dynamic_mask].copy() if non_dynamic_mask.any() else pd.DataFrame()

    # Process dynamic points (bin and average)
    if dynamic_mask.any():
        df_dynamic_binned = create_binned_data(df[dynamic_mask], bin_size)
        n_bins = len(df_dynamic_binned)
        print(f"Dynamic data: {sum(dynamic_mask)} points averaged into {n_bins} bins of {bin_size} points each")
    else:
        df_dynamic_binned = pd.DataFrame()
        print("No dynamic data found")

    if not df_non_dynamic.empty:
        print(f"Non-dynamic points kept: {len(df_non_dynamic)}")

    # Calculate statistics from original full dataset for 2-std area
    weight_mean = df['Scale Weight [g]'].mean()
    weight_std = df['Scale Weight [g]'].std()
    upper_2std = weight_mean + 2 * weight_std
    lower_2std = weight_mean - 2 * weight_std

    # Create the interactive plot
    fig = go.Figure()

    # Add 2-std area (use time range from all data)
    if not df_dynamic_binned.empty or not df_non_dynamic.empty:
        # Determine time range from available data
        time_min = df['Timestamp'].min()
        time_max = df['Timestamp'].max()
        time_range = [time_min, time_max]

        fig.add_trace(go.Scatter(
            x=time_range + time_range[::-1],
            y=[upper_2std, upper_2std, lower_2std, lower_2std],
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(color='rgba(128, 128, 128, 0)'),
            name='±2 Std Dev',
            hoverinfo='skip',
            showlegend=True
        ))

        # Add mean line (simplified)
        fig.add_trace(go.Scatter(
            x=time_range,
            y=[weight_mean, weight_mean],
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            name=f'Mean ({weight_mean:.4f} g)',
            hoverinfo='skip'
        ))

    # Determine scatter type based on use_webgl
    scatter_type = go.Scattergl if use_webgl else go.Scatter

    # Plot binned dynamic points with error bars
    if not df_dynamic_binned.empty:
        fig.add_trace(scatter_type(
            x=df_dynamic_binned['Timestamp'],
            y=df_dynamic_binned['Weight_mean'],
            error_y=dict(
                type='data',
                array=df_dynamic_binned['Weight_std'],
                visible=True,
                color='blue',
                thickness=1.5,
                width=3
            ),
            mode='markers',
            marker=dict(
                color='blue',
                size=6,
                opacity=0.8
            ),
            name=f'Dynamic (bins of {bin_size})',
            hovertemplate='<b>Dynamic (binned)</b><br>' +
                          'Time: %{x}<br>' +
                          'Weight: %{y:.4f} ± %{error_y.array:.4f} g<br>' +
                          f'Bin size: {bin_size} points<br>' +
                          '<extra></extra>',
            yaxis='y'
        ))

    # Plot individual non-dynamic points (if any)
    if not df_non_dynamic.empty:
        fig.add_trace(scatter_type(
            x=df_non_dynamic['Timestamp'],
            y=df_non_dynamic['Scale Weight [g]'],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                line=dict(width=2)
            ),
            name='Non-Dynamic',
            hovertemplate='<b>Non-Dynamic</b><br>' +
                          'Time: %{x}<br>' +
                          'Weight: %{y:.4f} g<br>' +
                          'Status: ' + df_non_dynamic['Scale Status'].astype(str) + '<br>' +
                          '<extra></extra>',
            yaxis='y'
        ))

    # Add binned TrueDyne MGCE1 Density on secondary y-axis with error bars
    if not df_dynamic_binned.empty:
        # Filter out NaN values for density data
        density_valid = df_dynamic_binned.dropna(subset=['Density_mean'])

        if not density_valid.empty:
            fig.add_trace(scatter_type(
                x=density_valid['Timestamp'],
                y=density_valid['Density_mean'],
                error_y=dict(
                    type='data',
                    array=density_valid['Density_std'],
                    visible=True,
                    color='green',
                    thickness=1,
                    width=2
                ),
                mode='markers',
                marker=dict(
                    color='green',
                    size=4,
                    opacity=0.7
                ),
                name=f'MGCE1 Density (bins of {bin_size})',
                yaxis='y2',
                hovertemplate='<b>MGCE1 Density (binned)</b><br>' +
                              'Time: %{x}<br>' +
                              'Density: %{y:.4f} ± %{error_y.array:.4f} kg/m³<br>' +
                              f'Bin size: {bin_size} points<br>' +
                              '<extra></extra>'
            ))

    # Customize the layout
    fig.update_layout(
        title={
            'text': f'Scale Weight & MGCE1 Density vs Timestamp (Binned: {bin_size} pts/bin)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Timestamp',
        yaxis=dict(
            title='Scale Weight [g]',
            side='left',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title=dict(
                text='TrueDyne MGCE1 Density [kg/m³]',
                font=dict(color='green')
            ),
            side='right',
            overlaying='y',
            showgrid=False,
            tickfont=dict(color='green')
        ),
        hovermode='closest',
        showlegend=True,
        width=1200,
        height=600,
        template='plotly_white',
        font=dict(size=12)
    )

    # Add grid for x-axis only (y-axis grid set above)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Show statistics (from original full dataset)
    print(f"\nStatistics from full dataset:")
    print(f"Total data points: {len(df)}")
    print(f"Dynamic points: {sum(df['Scale Status'] == 'Dynamic')}")
    print(f"Non-dynamic points: {sum(df['Scale Status'] != 'Dynamic')}")
    print(f"Unique Scale Status values: {df['Scale Status'].unique()}")

    # Weight statistics
    print(f"\nWeight statistics:")
    print(f"Min weight: {df['Scale Weight [g]'].min():.4f} g")
    print(f"Max weight: {df['Scale Weight [g]'].max():.4f} g")
    print(f"Mean weight: {weight_mean:.4f} g")
    print(f"Std deviation: {weight_std:.4f} g")
    print(f"2-std range: {lower_2std:.4f} g to {upper_2std:.4f} g")

    # Density statistics (from original data)
    density_col = 'TrueDyne MGCE1 Density [kg/m³]'
    density_valid = df[density_col].dropna()
    if not density_valid.empty:
        print(f"\nDensity statistics (original data):")
        print(f"Valid density points: {len(density_valid)} / {len(df)}")
        print(f"Min density: {density_valid.min():.4f} kg/m³")
        print(f"Max density: {density_valid.max():.4f} kg/m³")
        print(f"Mean density: {density_valid.mean():.4f} kg/m³")
        print(f"Std deviation: {density_valid.std():.4f} kg/m³")
    else:
        print(f"\nNo valid density data found in column '{density_col}'")

    # Binning statistics
    if not df_dynamic_binned.empty:
        print(f"\nBinning statistics:")
        print(f"Bin size: {bin_size} points per bin")
        print(f"Total bins created: {len(df_dynamic_binned)}")
        print(
            f"Data reduction: {len(df)} → {len(df_dynamic_binned)} points ({100 * (1 - len(df_dynamic_binned) / len(df)):.1f}% reduction)")
        print(f"Average weight std per bin: {df_dynamic_binned['Weight_std'].mean():.4f} g")
        if 'Density_std' in df_dynamic_binned.columns:
            avg_density_std = df_dynamic_binned['Density_std'].mean()
            print(f"Average density std per bin: {avg_density_std:.4f} kg/m³")

    # Correlation Analysis
    print(f"\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)

    # Original data correlation
    weight_col = 'Scale Weight [g]'
    density_col = 'TrueDyne MGCE1 Density [kg/m³]'

    # Filter data where both weight and density are available
    valid_data = df.dropna(subset=[weight_col, density_col])

    if len(valid_data) > 1:
        correlation_original = valid_data[weight_col].corr(valid_data[density_col])
        print(f"\nOriginal data correlation:")
        print(f"Data points with both measurements: {len(valid_data)} / {len(df)}")
        print(f"Pearson correlation coefficient: {correlation_original:.6f}")

        # Interpretation
        if abs(correlation_original) >= 0.8:
            strength = "very strong"
        elif abs(correlation_original) >= 0.6:
            strength = "strong"
        elif abs(correlation_original) >= 0.4:
            strength = "moderate"
        elif abs(correlation_original) >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if correlation_original > 0 else "negative"
        print(f"Interpretation: {strength} {direction} correlation")

        # Statistical significance (basic t-test)
        import scipy.stats as stats
        n = len(valid_data)
        if n > 2:
            # Calculate t-statistic for correlation
            t_stat = correlation_original * np.sqrt((n - 2) / (1 - correlation_original ** 2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            print(f"Statistical significance: p-value = {p_value:.6f}")
            if p_value < 0.001:
                print("Significance level: *** (p < 0.001) - highly significant")
            elif p_value < 0.01:
                print("Significance level: ** (p < 0.01) - very significant")
            elif p_value < 0.05:
                print("Significance level: * (p < 0.05) - significant")
            else:
                print("Significance level: ns (p ≥ 0.05) - not significant")
    else:
        print(f"\nInsufficient data for correlation analysis")
        print(f"Data points with both measurements: {len(valid_data)} / {len(df)}")

    # Binned data correlation (if available)
    if not df_dynamic_binned.empty:
        binned_valid = df_dynamic_binned.dropna(subset=['Weight_mean', 'Density_mean'])

        if len(binned_valid) > 1:
            correlation_binned = binned_valid['Weight_mean'].corr(binned_valid['Density_mean'])
            print(f"\nBinned data correlation:")
            print(f"Valid bins: {len(binned_valid)} / {len(df_dynamic_binned)}")
            print(f"Pearson correlation coefficient: {correlation_binned:.6f}")

            # Compare with original correlation
            if len(valid_data) > 1:
                diff = abs(correlation_binned - correlation_original)
                print(f"Difference from original: {correlation_binned - correlation_original:+.6f}")
                if diff < 0.01:
                    print("Binning preserves correlation very well")
                elif diff < 0.05:
                    print("Binning preserves correlation well")
                else:
                    print("Binning significantly changes correlation")

    # Time-dependent correlation analysis
    if len(valid_data) > 100:  # Only if enough data points
        print(f"\nTime-dependent correlation analysis:")

        # Calculate rolling correlation (using 1000-point windows)
        window_size = min(1000, len(valid_data) // 10)
        if window_size >= 50:
            try:
                # Create a temporary dataframe with both columns for rolling correlation
                temp_df = valid_data[[weight_col, density_col]].copy()
                rolling_corr = temp_df[weight_col].rolling(window=window_size).corr(temp_df[density_col]).dropna()

                if not rolling_corr.empty:
                    print(f"Rolling correlation (window: {window_size} points):")
                    print(f"  Mean: {rolling_corr.mean():.6f}")
                    print(f"  Std:  {rolling_corr.std():.6f}")
                    print(f"  Min:  {rolling_corr.min():.6f}")
                    print(f"  Max:  {rolling_corr.max():.6f}")

                    # Check for correlation stability
                    corr_range = rolling_corr.max() - rolling_corr.min()
                    if corr_range < 0.1:
                        print("  Stability: Very stable correlation over time")
                    elif corr_range < 0.3:
                        print("  Stability: Moderately stable correlation over time")
                    else:
                        print("  Stability: Correlation varies significantly over time")
                else:
                    print(f"Rolling correlation calculation produced no valid results")
            except Exception as e:
                print(f"Rolling correlation calculation failed: {e}")
                # Alternative: calculate correlation in chunks
                try:
                    chunk_size = window_size
                    correlations = []
                    for i in range(0, len(valid_data) - chunk_size, chunk_size // 2):
                        chunk = valid_data.iloc[i:i + chunk_size]
                        if len(chunk) >= 10:  # Minimum points for correlation
                            corr = chunk[weight_col].corr(chunk[density_col])
                            if not np.isnan(corr):
                                correlations.append(corr)

                    if correlations:
                        correlations = np.array(correlations)
                        print(f"Chunked correlation analysis ({len(correlations)} chunks of {chunk_size} points):")
                        print(f"  Mean: {correlations.mean():.6f}")
                        print(f"  Std:  {correlations.std():.6f}")
                        print(f"  Min:  {correlations.min():.6f}")
                        print(f"  Max:  {correlations.max():.6f}")

                        # Check for correlation stability
                        corr_range = correlations.max() - correlations.min()
                        if corr_range < 0.1:
                            print("  Stability: Very stable correlation over time")
                        elif corr_range < 0.3:
                            print("  Stability: Moderately stable correlation over time")
                        else:
                            print("  Stability: Correlation varies significantly over time")
                    else:
                        print("No valid correlations calculated in chunks")
                except Exception as e2:
                    print(f"Alternative correlation analysis also failed: {e2}")

    print("=" * 50)

    # Save as HTML file in the same directory as the CSV file (no interactive display)
    import os
    csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
    html_filename = os.path.join(csv_dir, "scale_weight_plot.html")

    # Configure for better performance
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'scale_weight_plot',
            'height': 600,
            'width': 1200,
            'scale': 1
        },
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }

    fig.write_html(html_filename, config=config)
    print(f"Plot saved as: {html_filename}")
    print(f"WebGL enabled: {use_webgl}")
    print(f"Bin size used: {bin_size}")

    return fig


if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250721_Messreihe_15\measurements_20250721_151542\measurements_20250721_151542\DMA_Results_20250721_151525_truedyne_buffer.csv"

    try:
        # Adjust these parameters:
        # bin_size: Number of consecutive points to average (try 50, 100, 200)
        # use_webgl: True = much faster for large datasets
        plot_scale_data(csv_file_path, bin_size=100, use_webgl=True)
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please update the csv_file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")