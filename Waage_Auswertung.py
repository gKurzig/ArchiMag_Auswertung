import pandas as pd
import plotly.graph_objects as go
import sys
import os


def find_weight_column(df):
    """
    Automatically detect the weight column from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to search

    Returns:
        str or None: The name of the weight column if found, None otherwise
    """
    possible_weight_columns = [
        'weight', 'Weight', 'WEIGHT',
        'mass', 'Mass', 'MASS',
        'Weight (mg)', 'Weight(mg)', 'Weight [mg]',
        'Weight (g)', 'Weight(g)', 'Weight [g]',
        'Mass (mg)', 'Mass(mg)', 'Mass [mg]',
        'Mass (g)', 'Mass(g)', 'Mass [g]'
    ]

    # First, try exact matches
    for col in df.columns:
        if col in possible_weight_columns:
            return col

    # Then, try partial matches
    for col in df.columns:
        if 'weight' in col.lower() or 'mass' in col.lower():
            return col

    return None


def load_data(csv_file_path):
    """
    Load the CSV file and return the DataFrame.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(df)} rows from {csv_file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def create_weight_plot(df, weight_column, output_file='weight_plot.html'):
    """
    Create a Plotly line plot of the weight data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data
        weight_column (str): Name of the weight column
        output_file (str): Output HTML file name

    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    fig = go.Figure()

    # Calculate statistics
    weight_mean = df[weight_column].mean()
    weight_std = df[weight_column].std()

    # Add weight trace
    fig.add_trace(go.Scatter(
        x=df['Relative Time [s]'],
        y=df[weight_column],
        mode='lines',
        name=f'Weight (μ={weight_mean:.3f}, σ={weight_std:.3f})',
        line=dict(color='blue', width=2)
    ))

    # Add horizontal line for average
    fig.add_hline(
        y=weight_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {weight_mean:.3f}",
        annotation_position="bottom right"
    )

    # Add shaded area for 2 sigma
    fig.add_hrect(
        y0=weight_mean - 2 * weight_std,
        y1=weight_mean + 2 * weight_std,
        fillcolor="lightblue",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="±2σ",
        annotation_position="top left"
    )

    # Update layout
    fig.update_layout(
        title='Weight vs Time',
        xaxis_title='Relative Time [s]',
        yaxis_title=f'{weight_column}',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )

    # Save the plot
    fig.write_html(output_file)
    print(f"Plot saved as '{output_file}'")

    return fig


def display_column_info(df):
    """
    Display information about available columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze
    """
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")

    print(f"\nDataFrame shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())


def plot_dma_weight(csv_file_path, weight_column=None, output_file='weight_plot.html'):
    """
    Main function to plot weight data from DMA CSV file.

    Args:
        csv_file_path (str): Path to the CSV file
        weight_column (str, optional): Name of the weight column. If None, auto-detect.
        output_file (str): Output HTML file name

    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Load the data
    df = load_data(csv_file_path)

    # Display basic info
    display_column_info(df)

    # Find weight column if not specified
    if weight_column is None:
        weight_column = find_weight_column(df)

    if weight_column is None:
        print("\nError: Could not automatically detect weight column.")
        print("Please specify the weight column manually by:")
        print("1. Using the column name: plot_dma_weight('file.csv', weight_column='COLUMN_NAME')")
        print("2. Or modify the script to use column index")
        return None

    # Validate that the column exists
    if weight_column not in df.columns:
        print(f"Error: Column '{weight_column}' not found in the DataFrame.")
        return None

    print(f"\nUsing column: '{weight_column}' for weight data")

    # Create and show the plot
    fig = create_weight_plot(df, weight_column, output_file)
    fig.show()

    return fig


def showWeight(df):
    """
    Function to analyze weight data from a DataFrame (used in correlation analysis).
    This function is called from the correlation analysis and works with the fused data.

    Args:
        df (pd.DataFrame): The DataFrame containing the fused data
    """
    print("\n=== Weight Analysis from Fused Data ===")

    # Try to find weight column in the fused data
    weight_column = find_weight_column(df)

    if weight_column is None:
        print("No weight column found in fused data.")
        return

    print(f"Found weight column: {weight_column}")

    # Create weight plot
    try:
        fig = create_weight_plot(df, weight_column, 'fused_data_weight_plot.html')
        if fig is not None:
            print("Weight plot from fused data created successfully!")
        else:
            print("Failed to create weight plot from fused data")
    except Exception as e:
        print(f"Error creating weight plot from fused data: {e}")


def main():
    """
    Main function to run the script standalone.
    """
    csv_file_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250715_Messreihe_11\measurements_20250715_172531\DMA_Results_20250715_172504_truedyne_buffer.csv"

    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found in current directory.")
        print("Please make sure the CSV file is in the same directory as this script.")
        sys.exit(1)

    # Plot the weight data
    fig = plot_dma_weight(csv_file_path)

    if fig is not None:
        print("\nPlot created successfully!")
        print("The plot has been displayed and saved as 'weight_plot.html'")
    else:
        print("\nFailed to create plot. Please check the error messages above.")


def plot_mgce1_reference_comparison(csv_path, rolling_window=100):
    """
    Create a plot using MGCE1 as reference instead of DMA.
    MGCE1 is the reference, DGFI1 is corrected to match MGCE1.
    All values have the MGCE1 mean subtracted to center around zero.
    Scale weight is plotted on a separate right y-axis.

    Args:
        csv_path: Path to the CSV file with truedyne data
        rolling_window: Window size for rolling average (default: 50)
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Read CSV file
    # Read CSV file with proper encoding
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')

    # Prepare data
    mgce1_reference = df['TrueDyne MGCE1 Density [kg/m³]']  # MGCE1 is reference
    dgfi1_test = df['TrueDyne DGFI1 Density [kg/m³]']
    scale_weight = df['Scale Weight [g]']
    time_data = pd.to_datetime(df['Timestamp'])

    # Remove NaN values for density comparison
    valid_mask = ~(pd.isna(mgce1_reference) | pd.isna(dgfi1_test))
    mgce1_clean = mgce1_reference[valid_mask]
    dgfi1_clean = dgfi1_test[valid_mask]
    time_clean = time_data[valid_mask]

    # Calculate MGCE1 mean for subtraction
    mgce1_mean = np.mean(mgce1_clean)

    # Calculate offset correction for DGFI1 (calibrate to MGCE1 reference)
    systematic_offset = np.mean(dgfi1_clean - mgce1_clean)
    dgfi1_corrected = dgfi1_clean - systematic_offset

    # Calculate residual statistics
    residuals = dgfi1_corrected - mgce1_clean
    residual_rms = np.sqrt(np.mean(residuals ** 2))
    correlation_coefficient = np.corrcoef(mgce1_clean, dgfi1_corrected)[0, 1]

    # Subtract MGCE1 mean from both MGCE1 and corrected DGFI1
    mgce1_centered = mgce1_clean - mgce1_mean
    dgfi1_centered = dgfi1_corrected - mgce1_mean

    # Calculate rolling averages for density data
    mgce1_rolling = pd.Series(mgce1_centered).rolling(window=rolling_window, center=True).mean()
    dgfi1_rolling = pd.Series(dgfi1_centered).rolling(window=rolling_window, center=True).mean()

    # Prepare scale weight data
    valid_scale_mask = ~pd.isna(scale_weight)
    scale_clean = scale_weight[valid_scale_mask]
    time_scale_clean = time_data[valid_scale_mask]
    scale_rolling = pd.Series(scale_clean).rolling(window=rolling_window, center=True).mean()

    # Calculate correlations with scale weight using rolling averages
    # MGCE1-Scale correlation
    common_times = pd.Index(time_clean).intersection(pd.Index(time_scale_clean))
    if len(common_times) > 1:
        mgce1_for_corr = mgce1_reference[time_data.isin(common_times)]
        scale_for_corr = scale_weight[time_data.isin(common_times)]
        # Remove any remaining NaN values
        valid_corr_mask = ~(pd.isna(mgce1_for_corr) | pd.isna(scale_for_corr))
        if np.sum(valid_corr_mask) > 1:
            # Calculate raw correlation first
            correlation_mgce1_scale_raw = np.corrcoef(mgce1_for_corr[valid_corr_mask],
                                                      scale_for_corr[valid_corr_mask])[0, 1]
            raw_corr_points = np.sum(valid_corr_mask)

            # Calculate rolling average correlation
            mgce1_roll_corr = pd.Series(mgce1_for_corr[valid_corr_mask]).rolling(window=rolling_window,
                                                                                 center=True).mean().dropna()
            scale_roll_corr = pd.Series(scale_for_corr[valid_corr_mask]).rolling(window=rolling_window,
                                                                                 center=True).mean().dropna()
            min_len = min(len(mgce1_roll_corr), len(scale_roll_corr))
            if min_len > 1:
                correlation_mgce1_scale = np.corrcoef(mgce1_roll_corr.iloc[:min_len],
                                                      scale_roll_corr.iloc[:min_len])[0, 1]
                rolling_corr_points = min_len
            else:
                correlation_mgce1_scale = np.nan
                rolling_corr_points = 0

            print(f"MGCE1-Scale Raw correlation: {correlation_mgce1_scale_raw:.4f} using {raw_corr_points} data points")
            print(
                f"MGCE1-Scale Rolling correlation: {correlation_mgce1_scale:.4f} using {rolling_corr_points} rolling avg points (from {len(mgce1_roll_corr)} values after rolling); Windowsize: {rolling_window}")
        else:
            correlation_mgce1_scale_raw = np.nan
            correlation_mgce1_scale = np.nan
            raw_corr_points = 0
            rolling_corr_points = 0
    else:
        correlation_mgce1_scale_raw = np.nan
        correlation_mgce1_scale = np.nan
        raw_corr_points = 0
        rolling_corr_points = 0

    # DGFI1-Scale correlation
    if len(common_times) > 1:
        dgfi1_for_corr = df['TrueDyne DGFI1 Density [kg/m³]'][time_data.isin(common_times)]
        valid_dgfi1_corr_mask = ~(pd.isna(dgfi1_for_corr) | pd.isna(scale_for_corr))
        if np.sum(valid_dgfi1_corr_mask) > 1:
            # Calculate raw correlation first
            correlation_dgfi1_scale_raw = np.corrcoef(dgfi1_for_corr[valid_dgfi1_corr_mask],
                                                      scale_for_corr[valid_dgfi1_corr_mask])[0, 1]
            raw_dgfi1_corr_points = np.sum(valid_dgfi1_corr_mask)

            # Calculate rolling average correlation
            dgfi1_roll_corr = pd.Series(dgfi1_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
                                                                                       center=True).mean().dropna()
            scale_roll_corr2 = pd.Series(scale_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
                                                                                        center=True).mean().dropna()
            min_len2 = min(len(dgfi1_roll_corr), len(scale_roll_corr2))
            if min_len2 > 1:
                correlation_dgfi1_scale = np.corrcoef(dgfi1_roll_corr.iloc[:min_len2],
                                                      scale_roll_corr2.iloc[:min_len2])[0, 1]
                rolling_dgfi1_corr_points = min_len2
            else:
                correlation_dgfi1_scale = np.nan
                rolling_dgfi1_corr_points = 0

            print(
                f"DGFI1-Scale Raw correlation: {correlation_dgfi1_scale_raw:.4f} using {raw_dgfi1_corr_points} data points; Windowsize: {rolling_window}")
            print(
                f"DGFI1-Scale Rolling correlation: {correlation_dgfi1_scale:.4f} using {rolling_dgfi1_corr_points} rolling avg points (from {len(dgfi1_roll_corr)} values after rolling)")
        else:
            correlation_dgfi1_scale_raw = np.nan
            correlation_dgfi1_scale = np.nan
            raw_dgfi1_corr_points = 0
            rolling_dgfi1_corr_points = 0
    else:
        correlation_dgfi1_scale_raw = np.nan
        correlation_dgfi1_scale = np.nan
        raw_dgfi1_corr_points = 0
        rolling_dgfi1_corr_points = 0

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add MGCE1 reference (centered) - raw data
    fig.add_trace(go.Scatter(
        x=time_clean, y=mgce1_centered,
        mode='lines', name='MGCE1 Reference (centered)',
        line=dict(color='red', width=1, dash='dot'),
        opacity=0.5
    ), secondary_y=False)

    # Add MGCE1 rolling average
    fig.add_trace(go.Scatter(
        x=time_clean, y=mgce1_rolling,
        mode='lines', name=f'MGCE1 Rolling Avg ({rolling_window} pts)',
        line=dict(color='red', width=2)
    ), secondary_y=False)

    # Add DGFI1 (calibrated and centered) - raw data
    fig.add_trace(go.Scatter(
        x=time_clean, y=dgfi1_centered,
        mode='lines', name='DGFI1 Calibrated (centered)',
        line=dict(color='blue', width=1, dash='dot'),
        opacity=0.5
    ), secondary_y=False)

    # Add DGFI1 rolling average
    fig.add_trace(go.Scatter(
        x=time_clean, y=dgfi1_rolling,
        mode='lines', name=f'DGFI1 Rolling Avg ({rolling_window} pts)',
        line=dict(color='blue', width=2)
    ), secondary_y=False)

    # Add scale weight data on secondary y-axis
    fig.add_trace(go.Scatter(
        x=time_scale_clean, y=scale_clean,
        mode='lines', name='Scale Weight',
        line=dict(color='orange', width=1, dash='dot'),
        opacity=0.5,
        yaxis='y2'
    ), secondary_y=True)

    # Add scale weight rolling average
    fig.add_trace(go.Scatter(
        x=time_scale_clean, y=scale_rolling,
        mode='lines', name=f'Scale Weight Rolling Avg ({rolling_window} pts)',
        line=dict(color='orange', width=2),
        yaxis='y2'
    ), secondary_y=True)

    # Add zero line for primary y-axis
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7)

    # Set y-axes titles
    fig.update_yaxes(title_text="Density Deviation from MGCE1 Mean [kg/m³]", secondary_y=False)
    fig.update_yaxes(title_text="Scale Weight [g]", secondary_y=True)

    # Update layout
    fig.update_layout(
        title='TrueDyne DGFI1 Calibrated to MGCE1 Reference (MGCE1 Mean Subtracted)',
        height=600,
        template='plotly_white',
        xaxis_title="Time"
    )

    # Add text box with statistics (showing rolling correlations)
    bias_direction = "higher" if systematic_offset > 0 else "lower"
    stats_text = (f"MGCE1 Mean: {mgce1_mean:.3f} kg/m³<br>"
                  f"DGFI1 bias: {systematic_offset:.3f} kg/m³ ({bias_direction})<br>"
                  f"After calibration:<br>"
                  f"Residual RMS: {residual_rms:.3f} kg/m³<br>"
                  f"MGCE1-DGFI1 Correlation: {correlation_coefficient:.4f}<br>"
                  f"<br>Rolling Avg Correlations ({rolling_window} pts):<br>"
                  f"MGCE1-Scale: {correlation_mgce1_scale:.4f} ({rolling_corr_points} pts)<br>"
                  f"DGFI1-Scale: {correlation_dgfi1_scale:.4f} ({rolling_dgfi1_corr_points} pts)")

    fig.add_annotation(
        x=0.02, y=0.98,
        text=stats_text,
        showarrow=False,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    # Save plot in same folder as input file
    import os
    input_dir = os.path.dirname(csv_path)
    filename = os.path.join(input_dir, "truedyne_mgce1_reference_calibration_with_scale.html")
    fig.write_html(filename)
    print(f"Saved: {filename}")

    return fig
# def plot_mgce1_reference_comparison(csv_path, rolling_window=50):
#     """
#     Create a plot using MGCE1 as reference instead of DMA.
#     MGCE1 is the reference, DGFI1 is corrected to match MGCE1.
#     All values have the MGCE1 mean subtracted to center around zero.
#     Scale weight is plotted on a separate right y-axis.
#
#     Args:
#         csv_path: Path to the CSV file with truedyne data
#         rolling_window: Window size for rolling average (default: 50)
#     """
#     import pandas as pd
#     import numpy as np
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#
#     # Read CSV file
#     # Read CSV file with proper encoding
#     try:
#         df = pd.read_csv(csv_path, encoding='utf-8')
#     except UnicodeDecodeError:
#         try:
#             df = pd.read_csv(csv_path, encoding='latin-1')
#         except UnicodeDecodeError:
#             df = pd.read_csv(csv_path, encoding='cp1252')
#
#     # Prepare data
#     mgce1_reference = df['TrueDyne MGCE1 Density [kg/m³]']  # MGCE1 is reference
#     dgfi1_test = df['TrueDyne DGFI1 Density [kg/m³]']
#     scale_weight = df['Scale Weight [g]']
#     time_data = pd.to_datetime(df['Timestamp'])
#
#     # Remove NaN values for density comparison
#     valid_mask = ~(pd.isna(mgce1_reference) | pd.isna(dgfi1_test))
#     mgce1_clean = mgce1_reference[valid_mask]
#     dgfi1_clean = dgfi1_test[valid_mask]
#     time_clean = time_data[valid_mask]
#
#     # Calculate MGCE1 mean for subtraction
#     mgce1_mean = np.mean(mgce1_clean)
#
#     # Calculate offset correction for DGFI1 (calibrate to MGCE1 reference)
#     systematic_offset = np.mean(dgfi1_clean - mgce1_clean)
#     dgfi1_corrected = dgfi1_clean - systematic_offset
#
#     # Calculate residual statistics
#     residuals = dgfi1_corrected - mgce1_clean
#     residual_rms = np.sqrt(np.mean(residuals ** 2))
#     correlation_coefficient = np.corrcoef(mgce1_clean, dgfi1_corrected)[0, 1]
#
#     # Subtract MGCE1 mean from both MGCE1 and corrected DGFI1
#     mgce1_centered = mgce1_clean - mgce1_mean
#     dgfi1_centered = dgfi1_corrected - mgce1_mean
#
#     # Calculate rolling averages for density data
#     mgce1_rolling = pd.Series(mgce1_centered).rolling(window=rolling_window, center=True).mean()
#     dgfi1_rolling = pd.Series(dgfi1_centered).rolling(window=rolling_window, center=True).mean()
#
#     # Prepare scale weight data
#     valid_scale_mask = ~pd.isna(scale_weight)
#     scale_clean = scale_weight[valid_scale_mask]
#     time_scale_clean = time_data[valid_scale_mask]
#     scale_rolling = pd.Series(scale_clean).rolling(window=rolling_window, center=True).mean()
#
#     # Calculate correlations with scale weight using rolling averages
#     # MGCE1-Scale correlation
#     common_times = pd.Index(time_clean).intersection(pd.Index(time_scale_clean))
#     if len(common_times) > 1:
#         mgce1_for_corr = mgce1_reference[time_data.isin(common_times)]
#         scale_for_corr = scale_weight[time_data.isin(common_times)]
#         # Remove any remaining NaN values
#         valid_corr_mask = ~(pd.isna(mgce1_for_corr) | pd.isna(scale_for_corr))
#         if np.sum(valid_corr_mask) > 1:
#             # Calculate raw correlation first
#             correlation_mgce1_scale_raw = np.corrcoef(mgce1_for_corr[valid_corr_mask],
#                                                       scale_for_corr[valid_corr_mask])[0, 1]
#             raw_corr_points = np.sum(valid_corr_mask)
#
#             # Calculate rolling average correlation
#             mgce1_roll_corr = pd.Series(mgce1_for_corr[valid_corr_mask]).rolling(window=rolling_window,
#                                                                                  center=True).mean().dropna()
#             scale_roll_corr = pd.Series(scale_for_corr[valid_corr_mask]).rolling(window=rolling_window,
#                                                                                  center=True).mean().dropna()
#             min_len = min(len(mgce1_roll_corr), len(scale_roll_corr))
#             if min_len > 1:
#                 correlation_mgce1_scale = np.corrcoef(mgce1_roll_corr.iloc[:min_len],
#                                                       scale_roll_corr.iloc[:min_len])[0, 1]
#                 rolling_corr_points = min_len
#             else:
#                 correlation_mgce1_scale = np.nan
#                 rolling_corr_points = 0
#
#             print(f"MGCE1-Scale Raw correlation: {correlation_mgce1_scale_raw:.4f} using {raw_corr_points} data points")
#             print(
#                 f"MGCE1-Scale Rolling correlation: {correlation_mgce1_scale:.4f} using {rolling_corr_points} rolling avg points")
#         else:
#             correlation_mgce1_scale_raw = np.nan
#             correlation_mgce1_scale = np.nan
#             raw_corr_points = 0
#             rolling_corr_points = 0
#     else:
#         correlation_mgce1_scale_raw = np.nan
#         correlation_mgce1_scale = np.nan
#         raw_corr_points = 0
#         rolling_corr_points = 0
#
#     # DGFI1-Scale correlation
#     if len(common_times) > 1:
#         dgfi1_for_corr = df['TrueDyne DGFI1 Density [kg/m³]'][time_data.isin(common_times)]
#         valid_dgfi1_corr_mask = ~(pd.isna(dgfi1_for_corr) | pd.isna(scale_for_corr))
#         if np.sum(valid_dgfi1_corr_mask) > 1:
#             # Calculate raw correlation first
#             correlation_dgfi1_scale_raw = np.corrcoef(dgfi1_for_corr[valid_dgfi1_corr_mask],
#                                                       scale_for_corr[valid_dgfi1_corr_mask])[0, 1]
#             raw_dgfi1_corr_points = np.sum(valid_dgfi1_corr_mask)
#
#             # Calculate rolling average correlation
#             dgfi1_roll_corr = pd.Series(dgfi1_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
#                                                                                        center=True).mean().dropna()
#             scale_roll_corr2 = pd.Series(scale_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
#                                                                                         center=True).mean().dropna()
#             min_len2 = min(len(dgfi1_roll_corr), len(scale_roll_corr2))
#             if min_len2 > 1:
#                 correlation_dgfi1_scale = np.corrcoef(dgfi1_roll_corr.iloc[:min_len2],
#                                                       scale_roll_corr2.iloc[:min_len2])[0, 1]
#                 rolling_dgfi1_corr_points = min_len2
#             else:
#                 correlation_dgfi1_scale = np.nan
#                 rolling_dgfi1_corr_points = 0
#
#             print(
#                 f"DGFI1-Scale Raw correlation: {correlation_dgfi1_scale_raw:.4f} using {raw_dgfi1_corr_points} data points")
#             print(
#                 f"DGFI1-Scale Rolling correlation: {correlation_dgfi1_scale:.4f} using {rolling_dgfi1_corr_points} rolling avg points")
#         else:
#             correlation_dgfi1_scale_raw = np.nan
#             correlation_dgfi1_scale = np.nan
#             raw_dgfi1_corr_points = 0
#             rolling_dgfi1_corr_points = 0
#     else:
#         correlation_dgfi1_scale_raw = np.nan
#         correlation_dgfi1_scale = np.nan
#         raw_dgfi1_corr_points = 0
#         rolling_dgfi1_corr_points = 0
#
#     # Create subplot with secondary y-axis
#     fig = make_subplots(specs=[[{"secondary_y": True}]])
#
#     # Add MGCE1 reference (centered) - raw data
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=mgce1_centered,
#         mode='lines', name='MGCE1 Reference (centered)',
#         line=dict(color='red', width=1, dash='dot'),
#         opacity=0.5
#     ), secondary_y=False)
#
#     # Add MGCE1 rolling average
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=mgce1_rolling,
#         mode='lines', name=f'MGCE1 Rolling Avg ({rolling_window} pts)',
#         line=dict(color='red', width=2)
#     ), secondary_y=False)
#
#     # Add DGFI1 (calibrated and centered) - raw data
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=dgfi1_centered,
#         mode='lines', name='DGFI1 Calibrated (centered)',
#         line=dict(color='blue', width=1, dash='dot'),
#         opacity=0.5
#     ), secondary_y=False)
#
#     # Add DGFI1 rolling average
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=dgfi1_rolling,
#         mode='lines', name=f'DGFI1 Rolling Avg ({rolling_window} pts)',
#         line=dict(color='blue', width=2)
#     ), secondary_y=False)
#
#     # Add scale weight data on secondary y-axis
#     fig.add_trace(go.Scatter(
#         x=time_scale_clean, y=scale_clean,
#         mode='lines', name='Scale Weight',
#         line=dict(color='orange', width=1, dash='dot'),
#         opacity=0.5,
#         yaxis='y2'
#     ), secondary_y=True)
#
#     # Add scale weight rolling average
#     fig.add_trace(go.Scatter(
#         x=time_scale_clean, y=scale_rolling,
#         mode='lines', name=f'Scale Weight Rolling Avg ({rolling_window} pts)',
#         line=dict(color='orange', width=2),
#         yaxis='y2'
#     ), secondary_y=True)
#
#     # Add zero line for primary y-axis
#     fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7)
#
#     # Set y-axes titles
#     fig.update_yaxes(title_text="Density Deviation from MGCE1 Mean [kg/m³]", secondary_y=False)
#     fig.update_yaxes(title_text="Scale Weight [g]", secondary_y=True)
#
#     # Update layout
#     fig.update_layout(
#         title='TrueDyne DGFI1 Calibrated to MGCE1 Reference (MGCE1 Mean Subtracted)',
#         height=600,
#         template='plotly_white',
#         xaxis_title="Time"
#     )
#
#     # Add text box with statistics (showing rolling correlations)
#     bias_direction = "higher" if systematic_offset > 0 else "lower"
#     stats_text = (f"MGCE1 Mean: {mgce1_mean:.3f} kg/m³<br>"
#                   f"DGFI1 bias: {systematic_offset:.3f} kg/m³ ({bias_direction})<br>"
#                   f"After calibration:<br>"
#                   f"Residual RMS: {residual_rms:.3f} kg/m³<br>"
#                   f"MGCE1-DGFI1 Correlation: {correlation_coefficient:.4f}<br>"
#                   f"<br>Rolling Avg Correlations ({rolling_window} pts):<br>"
#                   f"MGCE1-Scale: {correlation_mgce1_scale:.4f} ({rolling_corr_points} pts)<br>"
#                   f"DGFI1-Scale: {correlation_dgfi1_scale:.4f} ({rolling_dgfi1_corr_points} pts)")
#
#     fig.add_annotation(
#         x=0.02, y=0.98,
#         text=stats_text,
#         showarrow=False,
#         xref="paper", yref="paper",
#         xanchor="left", yanchor="top",
#         bgcolor="rgba(255,255,255,0.8)",
#         bordercolor="black",
#         borderwidth=1
#     )
#
#     # Save plot
#     filename = "truedyne_mgce1_reference_calibration_with_scale.html"
#     fig.write_html(filename)
#     print(f"Saved: {filename}")
#
#     return fig



# #########################################################################
# def plot_mgce1_reference_comparison_old(csv_path, rolling_window=50):
#     """
#     Create a plot using MGCE1 as reference instead of DMA.
#     MGCE1 is the reference, DGFI1 is corrected to match MGCE1.
#     All values have the MGCE1 mean subtracted to center around zero.
#     Scale weight is plotted on a separate right y-axis.
#
#     Args:
#         csv_path: Path to the CSV file with truedyne data
#         rolling_window: Window size for rolling average (default: 50)
#     """
#     import pandas as pd
#     import numpy as np
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#
#     # Read CSV file
#     # Read CSV file with proper encoding
#     try:
#         df = pd.read_csv(csv_path, encoding='utf-8')
#     except UnicodeDecodeError:
#         try:
#             df = pd.read_csv(csv_path, encoding='latin-1')
#         except UnicodeDecodeError:
#             df = pd.read_csv(csv_path, encoding='cp1252')
#
#     # Prepare data
#     mgce1_reference = df['TrueDyne MGCE1 Density [kg/m³]']  # MGCE1 is reference
#     dgfi1_test = df['TrueDyne DGFI1 Density [kg/m³]']
#     scale_weight = df['Scale Weight [g]']
#     time_data = pd.to_datetime(df['Timestamp'])
#
#     # Remove NaN values for density comparison
#     valid_mask = ~(pd.isna(mgce1_reference) | pd.isna(dgfi1_test))
#     mgce1_clean = mgce1_reference[valid_mask]
#     dgfi1_clean = dgfi1_test[valid_mask]
#     time_clean = time_data[valid_mask]
#
#     # Calculate MGCE1 mean for subtraction
#     mgce1_mean = np.mean(mgce1_clean)
#
#     # Calculate offset correction for DGFI1 (calibrate to MGCE1 reference)
#     systematic_offset = np.mean(dgfi1_clean - mgce1_clean)
#     dgfi1_corrected = dgfi1_clean - systematic_offset
#
#     # Calculate residual statistics
#     residuals = dgfi1_corrected - mgce1_clean
#     residual_rms = np.sqrt(np.mean(residuals ** 2))
#     correlation_coefficient = np.corrcoef(mgce1_clean, dgfi1_corrected)[0, 1]
#
#     # Subtract MGCE1 mean from both MGCE1 and corrected DGFI1
#     mgce1_centered = mgce1_clean - mgce1_mean
#     dgfi1_centered = dgfi1_corrected - mgce1_mean
#
#     # Calculate rolling averages for density data
#     mgce1_rolling = pd.Series(mgce1_centered).rolling(window=rolling_window, center=True).mean()
#     dgfi1_rolling = pd.Series(dgfi1_centered).rolling(window=rolling_window, center=True).mean()
#
#     # Prepare scale weight data
#     valid_scale_mask = ~pd.isna(scale_weight)
#     scale_clean = scale_weight[valid_scale_mask]
#     time_scale_clean = time_data[valid_scale_mask]
#     scale_rolling = pd.Series(scale_clean).rolling(window=rolling_window, center=True).mean()
#
#     # Calculate correlations with scale weight
#     # MGCE1-Scale correlation
#     common_times = pd.Index(time_clean).intersection(pd.Index(time_scale_clean))
#     if len(common_times) > 1:
#         mgce1_for_corr = mgce1_reference[time_data.isin(common_times)]
#         scale_for_corr = scale_weight[time_data.isin(common_times)]
#         # Remove any remaining NaN values
#         valid_corr_mask = ~(pd.isna(mgce1_for_corr) | pd.isna(scale_for_corr))
#         if np.sum(valid_corr_mask) > 1:
#             # Get rolling averages for correlation points
#             mgce1_roll_corr = pd.Series(mgce1_for_corr[valid_corr_mask]).rolling(window=rolling_window,
#                                                                                  center=True).mean().dropna()
#             scale_roll_corr = pd.Series(scale_for_corr[valid_corr_mask]).rolling(window=rolling_window,
#                                                                                  center=True).mean().dropna()
#             min_len = min(len(mgce1_roll_corr), len(scale_roll_corr))
#             correlation_mgce1_scale = np.corrcoef(mgce1_roll_corr.iloc[:min_len],
#                                                   scale_roll_corr.iloc[:min_len])[0, 1]
#
#             print(f"MGCE1-Scale correlation calculated using {np.sum(valid_corr_mask)} paired data points")
#         else:
#             correlation_mgce1_scale = np.nan
#     else:
#         correlation_mgce1_scale = np.nan
#
#     # DGFI1-Scale correlation
#     if len(common_times) > 1:
#         dgfi1_for_corr = df['TrueDyne DGFI1 Density [kg/m³]'][time_data.isin(common_times)]
#         valid_dgfi1_corr_mask = ~(pd.isna(dgfi1_for_corr) | pd.isna(scale_for_corr))
#         if np.sum(valid_dgfi1_corr_mask) > 1:
#             # Get rolling averages for correlation points
#             dgfi1_roll_corr = pd.Series(dgfi1_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
#                                                                                        center=True).mean().dropna()
#             scale_roll_corr2 = pd.Series(scale_for_corr[valid_dgfi1_corr_mask]).rolling(window=rolling_window,
#                                                                                         center=True).mean().dropna()
#             min_len2 = min(len(dgfi1_roll_corr), len(scale_roll_corr2))
#             correlation_dgfi1_scale = np.corrcoef(dgfi1_roll_corr.iloc[:min_len2],
#                                                   scale_roll_corr2.iloc[:min_len2])[0, 1]
#             print(f"DGFI1-Scale correlation calculated using {np.sum(valid_dgfi1_corr_mask)} paired data points")
#         else:
#             correlation_dgfi1_scale = np.nan
#     else:
#         correlation_dgfi1_scale = np.nan
#
#     # Create subplot with secondary y-axis
#     fig = make_subplots(specs=[[{"secondary_y": True}]])
#
#     # Add MGCE1 reference (centered) - raw data
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=mgce1_centered,
#         mode='lines', name='MGCE1 Reference (centered)',
#         line=dict(color='red', width=1, dash='dot'),
#         opacity=0.5
#     ), secondary_y=False)
#
#     # Add MGCE1 rolling average
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=mgce1_rolling,
#         mode='lines', name=f'MGCE1 Rolling Avg ({rolling_window} pts)',
#         line=dict(color='red', width=2)
#     ), secondary_y=False)
#
#     # Add DGFI1 (calibrated and centered) - raw data
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=dgfi1_centered,
#         mode='lines', name='DGFI1 Calibrated (centered)',
#         line=dict(color='blue', width=1, dash='dot'),
#         opacity=0.5
#     ), secondary_y=False)
#
#     # Add DGFI1 rolling average
#     fig.add_trace(go.Scatter(
#         x=time_clean, y=dgfi1_rolling,
#         mode='lines', name=f'DGFI1 Rolling Avg ({rolling_window} pts)',
#         line=dict(color='blue', width=2)
#     ), secondary_y=False)
#
#     # Add scale weight data on secondary y-axis
#     fig.add_trace(go.Scatter(
#         x=time_scale_clean, y=scale_clean,
#         mode='lines', name='Scale Weight',
#         line=dict(color='orange', width=1, dash='dot'),
#         opacity=0.5,
#         yaxis='y2'
#     ), secondary_y=True)
#
#     # Add scale weight rolling average
#     fig.add_trace(go.Scatter(
#         x=time_scale_clean, y=scale_rolling,
#         mode='lines', name=f'Scale Weight Rolling Avg ({rolling_window} pts)',
#         line=dict(color='orange', width=2),
#         yaxis='y2'
#     ), secondary_y=True)
#
#     # Add zero line for primary y-axis
#     fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7)
#
#     # Set y-axes titles
#     fig.update_yaxes(title_text="Density Deviation from MGCE1 Mean [kg/m³]", secondary_y=False)
#     fig.update_yaxes(title_text="Scale Weight [g]", secondary_y=True)
#
#     # Update layout
#     fig.update_layout(
#         title='TrueDyne DGFI1 Calibrated to MGCE1 Reference (MGCE1 Mean Subtracted)',
#         height=600,
#         template='plotly_white',
#         xaxis_title="Time"
#     )
#
#     # Add text box with statistics
#     bias_direction = "higher" if systematic_offset > 0 else "lower"
#     stats_text = (f"MGCE1 Mean: {mgce1_mean:.3f} kg/m³<br>"
#                   f"DGFI1 bias: {systematic_offset:.3f} kg/m³ ({bias_direction})<br>"
#                   f"After calibration:<br>"
#                   f"Residual RMS: {residual_rms:.3f} kg/m³<br>"
#                   f"MGCE1-DGFI1 Correlation: {correlation_coefficient:.4f}<br>"
#                   f"<br>MGCE1-Scale Correlation: {correlation_mgce1_scale:.4f}<br>"
#                   f"DGFI1-Scale Correlation: {correlation_dgfi1_scale:.4f}")
#
#     fig.add_annotation(
#         x=0.02, y=0.98,
#         text=stats_text,
#         showarrow=False,
#         xref="paper", yref="paper",
#         xanchor="left", yanchor="top",
#         bgcolor="rgba(255,255,255,0.8)",
#         bordercolor="black",
#         borderwidth=1
#     )
#
#     # Save plot
#     filename = "truedyne_mgce1_reference_calibration_with_scale.html"
#     fig.write_html(filename)
#     print(f"Saved: {filename}")
#     print(f"MGCE1-Scale Weight Correlation: {correlation_mgce1_scale:.4f}")
#     print(f"DGFI1-Scale Weight Correlation: {correlation_dgfi1_scale:.4f}")
#
#     return fig

if __name__ == "__main__":
    # Replace with your actual file path
    fig = plot_mgce1_reference_comparison(r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250721_Messreihe_15\measurements_20250721_151542\measurements_20250721_151542\DMA_Results_20250721_151525_truedyne_buffer.csv")

