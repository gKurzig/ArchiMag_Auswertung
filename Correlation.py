import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import seaborn as sns


def add_weather_data(df, weather_filename='weather_data.csv'):
    """
    Add weather data (temperature and pressure) to the measurement dataframe.
    Matches timestamps and adds env_temp and env_pres columns.
    """
    print("Loading weather data...")
    try:
        # Load weather data
        weather_df = pd.read_csv(weather_filename)

        # Convert weather timestamp to datetime
        weather_df['time'] = pd.to_datetime(weather_df['time'])

        # Rename columns for clarity
        weather_df = weather_df.rename(columns={
            'time': 'weather_time',
            'temp': 'env_temp',
            'pres': 'env_pres'
        })

        # Select only needed columns
        weather_df = weather_df[['weather_time', 'env_temp', 'env_pres']]

        # Fix timezone issues - convert both to timezone-naive
        df_tz_naive = df.copy()
        df_tz_naive['Timestamp'] = df_tz_naive['Timestamp'].dt.tz_localize(None)

        # Round timestamps to nearest hour for matching (using 'h' instead of 'H')
        df_tz_naive['timestamp_rounded'] = df_tz_naive['Timestamp'].dt.round('h')
        weather_df['timestamp_rounded'] = weather_df['weather_time'].dt.round('h')

        # Merge dataframes on rounded timestamps
        df_merged = pd.merge(
            df_tz_naive,
            weather_df[['timestamp_rounded', 'env_temp', 'env_pres']],
            on='timestamp_rounded',
            how='left'  # Keep all measurement data, add weather where available
        )

        # Drop the temporary rounded timestamp column
        df_merged = df_merged.drop('timestamp_rounded', axis=1)

        # Restore original timestamp (with timezone info) for final dataframe
        df_merged['Timestamp'] = df['Timestamp']

        # Count successful matches
        matched_count = df_merged['env_temp'].notna().sum()
        total_count = len(df_merged)

        print(f"Weather data loaded: {matched_count}/{total_count} measurements matched with weather data")
        print(f"Weather data range: {weather_df['weather_time'].min()} to {weather_df['weather_time'].max()}")

        # Show some statistics
        if matched_count > 0:
            print(
                f"Environmental temperature range: {df_merged['env_temp'].min():.1f}°C to {df_merged['env_temp'].max():.1f}°C")
            print(
                f"Environmental pressure range: {df_merged['env_pres'].min():.1f} to {df_merged['env_pres'].max():.1f} hPa")

        return df_merged

    except Exception as e:
        print(f"Error loading weather data: {e}")
        print("Continuing without weather data...")
        # Add empty columns if weather loading fails
        df['env_temp'] = np.nan
        df['env_pres'] = np.nan
        return df

def load_and_prepare_data(filename='fused_data_3.csv'):
    """Load and prepare the data for analysis."""
    # Load the CSV data
    df = pd.read_csv(filename)

    # Clean column names for easier handling
    df.columns = df.columns.str.strip()

    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert DMA density from g/cm³ to kg/m³ (multiply by 1000)
    df['Density'] = df['Density'] * 1000


    # Add weather data
    df = add_weather_data(df)

    # Create shorter column names for better visualization
    column_mapping = {
        'T(cell)': 'Cell_Temp',
        'Density': 'Density',
        'TrueDyne MGCE1 Density (Average)': 'MGCE1_Density',
        'TrueDyne MGCE1 Press (Average)': 'MGCE1_Pressure',
        'TrueDyne MGCE1 Temp (Average)': 'MGCE1_Temp',
        'TrueDyne DGFI1 Density (Average)': 'DGFI1_Density',
        'TrueDyne DGFI1 Press (Average)': 'DGFI1_Pressure',
        'TrueDyne DGFI1 Temp (Average)': 'DGFI1_Temp',
        'env_temp': 'Env_Temp',
        'env_pres': 'Env_Pressure'
    }


    # Create a DataFrame with renamed columns for analysis
    df_analysis = df[list(column_mapping.keys())].rename(columns=column_mapping)

    # Select numerical columns for correlation analysis
    numerical_cols = ['Cell_Temp', 'Density', 'MGCE1_Density', 'MGCE1_Pressure', 'MGCE1_Temp',
                      'DGFI1_Density', 'DGFI1_Pressure', 'DGFI1_Temp']

    return df, df_analysis, numerical_cols


def print_data_overview(df, df_analysis):
    """Print basic overview of the dataset."""
    print("Dataset Overview:")
    print(f"Shape: {df_analysis.shape}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print("\nBasic Statistics:")
    print(df_analysis.describe())


def create_correlation_heatmap(correlation_matrix):
    """Create and save correlation heatmap."""
    print("Creating correlation heatmap...")
    try:
        fig_heatmap = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of All Measurements",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig_heatmap.update_layout(
            width=800, height=600,
            title_x=0.5,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        fig_heatmap.write_html("correlation_heatmap.html")
        print("Heatmap saved as correlation_heatmap.html")
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        print("Continuing with text-based correlation matrix:")
        print(correlation_matrix)


def create_scatter_matrix(df_analysis):
    """Create scatter matrix for key variables."""
    print("Creating scatter matrix...")
    try:
        #key_vars = ['Cell_Temp', 'MGCE1_Density', 'DGFI1_Density', 'MGCE1_Temp', 'DGFI1_Temp']
        key_vars = ['Cell_Temp', 'MGCE1_Density', 'DGFI1_Density', 'MGCE1_Pressure', 'DGFI1_Pressure', 'MGCE1_Temp',
                    'DGFI1_Temp']
        fig_scatter = px.scatter_matrix(
            df_analysis,
            dimensions=key_vars,
            title="Scatter Matrix of Key Variables",
            width=1000, height=800
        )
        fig_scatter.update_traces(diagonal_visible=False)
        fig_scatter.write_html("scatter_matrix.html")
        print("Scatter matrix saved as scatter_matrix.html")
    except Exception as e:
        print(f"Error creating scatter matrix: {e}")
################################################################
def create_density_timeseries_plot(df, df_analysis, window_size=50):
    """Create time series plot of all density measurements with rolling averages."""
    print("Creating density time series plot...")
    try:
        # Calculate rolling averages for all density measurements
        df_analysis['DMA_Density_Rolling'] = df_analysis['Density'].rolling(window=window_size, center=True).mean()
        df_analysis['MGCE1_Density_Rolling'] = df_analysis['MGCE1_Density'].rolling(window=window_size,
                                                                                    center=True).mean()
        df_analysis['DGFI1_Density_Rolling'] = df_analysis['DGFI1_Density'].rolling(window=window_size,
                                                                                    center=True).mean()

        # Calculate rolling standard deviations for error bars
        df_analysis['DMA_Density_Rolling_Std'] = df_analysis['Density'].rolling(window=window_size, center=True).std()
        df_analysis['MGCE1_Density_Rolling_Std'] = df_analysis['MGCE1_Density'].rolling(window=window_size,
                                                                                        center=True).std()
        df_analysis['DGFI1_Density_Rolling_Std'] = df_analysis['DGFI1_Density'].rolling(window=window_size,
                                                                                        center=True).std()

        # Create the plot
        fig = go.Figure()

        # Add raw data lines (lighter/thinner)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Density'],
            mode='lines',
            name='DMA Density (raw)',
            line=dict(color='lightcoral', width=1),
            opacity=0.5
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Density'],
            mode='lines',
            name='MGCE1 Density (raw)',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Density'],
            mode='lines',
            name='DGFI1 Density (raw)',
            line=dict(color='lightgreen', width=1),
            opacity=0.5
        ))

        # Create error bar masks for every 100th point
        error_mask = np.arange(len(df_analysis)) % 100 == 0
        dma_error_array = np.where(error_mask, df_analysis['DMA_Density_Rolling_Std'], np.nan)
        mgce1_error_array = np.where(error_mask, df_analysis['MGCE1_Density_Rolling_Std'], np.nan)
        dgfi1_error_array = np.where(error_mask, df_analysis['DGFI1_Density_Rolling_Std'], np.nan)

        # Add rolling average lines with error bars (bolder)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_Density_Rolling'],
            error_y=dict(
                type='data',
                array=dma_error_array,
                visible=True,
                color='red',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DMA Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='red', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Density_Rolling'],
            error_y=dict(
                type='data',
                array=mgce1_error_array,
                visible=True,
                color='blue',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'MGCE1 Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='blue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Density_Rolling'],
            error_y=dict(
                type='data',
                array=dgfi1_error_array,
                visible=True,
                color='green',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DGFI1 Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='green', width=3)
        ))

        # Update layout
        fig.update_layout(
            title=f"Density Time Series with {window_size}-Point Rolling Averages",
            xaxis_title="Time",
            yaxis_title="Density (kg/m³)",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot
        fig.write_html("density_timeseries_with_rolling_avg.html")
        print("Density time series plot saved as density_timeseries_with_rolling_avg.html")

        # Print statistics
        print(f"\nDensity Statistics:")
        print(f"DMA Density:")
        print(f"  Mean: {df_analysis['Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['Density'].min():.6f}, {df_analysis['Density'].max():.6f}] kg/m³")

        print(f"\nMGCE1 Density:")
        print(f"  Mean: {df_analysis['MGCE1_Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['MGCE1_Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['MGCE1_Density'].min():.6f}, {df_analysis['MGCE1_Density'].max():.6f}] kg/m³")

        print(f"\nDGFI1 Density:")
        print(f"  Mean: {df_analysis['DGFI1_Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['DGFI1_Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['DGFI1_Density'].min():.6f}, {df_analysis['DGFI1_Density'].max():.6f}] kg/m³")

    except Exception as e:
        print(f"Error creating density time series plot: {e}")
#######################################################
def create_dma_truedyne_rolling_difference_plot(df, df_analysis, instrument='MGCE1', window_size=50):
    """Create plot showing the difference between DMA and specified TrueDyne instrument rolling averages."""
    print(f"Creating DMA - {instrument} rolling difference plot...")
    try:
        # Validate instrument parameter
        if instrument not in ['MGCE1', 'DGFI1']:
            raise ValueError("Instrument must be 'MGCE1' or 'DGFI1'")

        # Make sure rolling averages are calculated (in case this function is called independently)
        if 'DMA_Density_Rolling' not in df_analysis.columns:
            df_analysis['DMA_Density_Rolling'] = df_analysis['Density'].rolling(window=window_size, center=True).mean()
        if f'{instrument}_Density_Rolling' not in df_analysis.columns:
            df_analysis[f'{instrument}_Density_Rolling'] = df_analysis[f'{instrument}_Density'].rolling(
                window=window_size, center=True).mean()

        # Calculate the difference: DMA - TrueDyne (rolling averages)
        diff_column = f'DMA_{instrument}_Rolling_Diff'
        df_analysis[diff_column] = df_analysis['DMA_Density_Rolling'] - df_analysis[f'{instrument}_Density_Rolling']

        # Calculate rolling standard deviation of the difference
        std_column = f'DMA_{instrument}_Rolling_Diff_Std'
        df_analysis[std_column] = df_analysis[diff_column].rolling(window=window_size, center=True).std()

        # Create the plot
        fig = go.Figure()

        # Create error bar mask for every 100th point (shifted by 50)
        error_mask = (np.arange(len(df_analysis)) + 50) % 100 == 0
        diff_error_array = np.where(error_mask, df_analysis[std_column], np.nan)

        # Set colors based on instrument
        colors = {
            'MGCE1': {'line': 'blue', 'error': 'darkblue'},
            'DGFI1': {'line': 'green', 'error': 'darkgreen'}
        }

        # Add the difference line with error bars
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis[diff_column],
            error_y=dict(
                type='data',
                array=diff_error_array,
                visible=True,
                color=colors[instrument]['error'],
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DMA - {instrument} ({window_size}-pt rolling avg diff ± 1σ every 100pts)',
            line=dict(color=colors[instrument]['line'], width=3)
        ))


        # Add a horizontal line at zero for reference
        #fig.add_hline(y=0, line_dash="dash", line_color="gray",
        #              annotation_text="Zero Difference")

        # Update layout
        fig.update_layout(
            title=f"Difference Between DMA and {instrument} Rolling Averages ({window_size}-point window)",
            xaxis_title="Time",
            yaxis_title=f"Density Difference (DMA - {instrument}) [kg/m³]",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot with instrument-specific filename
        filename = f"dma_{instrument.lower()}_rolling_difference.html"
        fig.write_html(filename)
        print(f"DMA - {instrument} rolling difference plot saved as {filename}")

        # Print statistics
        mean_diff = df_analysis[diff_column].mean()
        std_diff = df_analysis[diff_column].std()
        min_diff = df_analysis[diff_column].min()
        max_diff = df_analysis[diff_column].max()

        print(f"\nDMA - {instrument} Rolling Average Difference Statistics:")
        print(f"  Mean difference: {mean_diff:.6f} kg/m³")
        print(f"  Std deviation:   {std_diff:.6f} kg/m³")
        print(f"  Range: [{min_diff:.6f}, {max_diff:.6f}] kg/m³")

        # Check if there's a systematic bias
        if abs(mean_diff) > 2 * std_diff:
            print(
                f"  → Systematic bias detected: DMA is consistently {'higher' if mean_diff > 0 else 'lower'} than {instrument}")
        else:
            print(f"  → No significant systematic bias detected")

    except Exception as e:
        print(f"Error creating DMA - {instrument} rolling difference plot: {e}")
#######################################################
def create_dma_total_avg_truedyne_rolling_difference_plot(df, df_analysis, instrument='MGCE1', window_size=50):
    """Create plot showing the difference between DMA total average and specified TrueDyne instrument rolling averages."""
    print(f"Creating DMA total avg - {instrument} rolling difference plot...")
    try:
        # Validate instrument parameter
        if instrument not in ['MGCE1', 'DGFI1']:
            raise ValueError("Instrument must be 'MGCE1' or 'DGFI1'")

        # Calculate DMA total average (single value)
        dma_total_avg = df_analysis['Density'].mean()
        print(f"DMA total average: {dma_total_avg:.6f} kg/m³")

        # Make sure TrueDyne rolling average is calculated
        if f'{instrument}_Density_Rolling' not in df_analysis.columns:
            df_analysis[f'{instrument}_Density_Rolling'] = df_analysis[f'{instrument}_Density'].rolling(
                window=window_size, center=True).mean()

        # Calculate the difference: DMA total avg - TrueDyne rolling average
        diff_column = f'DMA_TotalAvg_{instrument}_Rolling_Diff'
        df_analysis[diff_column] = dma_total_avg - df_analysis[f'{instrument}_Density_Rolling']

        # Calculate rolling standard deviation of the difference
        std_column = f'DMA_TotalAvg_{instrument}_Rolling_Diff_Std'
        df_analysis[std_column] = df_analysis[diff_column].rolling(window=window_size, center=True).std()

        # Create the plot
        fig = go.Figure()

        # Create error bar mask for every 100th point (shifted by 50)
        error_mask = (np.arange(len(df_analysis)) + 50) % 100 == 0
        diff_error_array = np.where(error_mask, df_analysis[std_column], np.nan)

        # Set colors based on instrument
        colors = {
            'MGCE1': {'line': 'blue', 'error': 'darkblue'},
            'DGFI1': {'line': 'green', 'error': 'darkgreen'}
        }

        # Add the difference line with error bars
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis[diff_column],
            error_y=dict(
                type='data',
                array=diff_error_array,
                visible=True,
                color=colors[instrument]['error'],
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DMA total avg - {instrument} rolling avg (diff ± 1σ every 100pts)',
            line=dict(color=colors[instrument]['line'], width=3)
        ))

        # Add DMA total average minus individual DMA values
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=dma_total_avg - df_analysis['DMA_Density_Rolling'],
            mode='lines',
            name='DMA Total Avg - DMA Individual Values',
            line=dict(color='lightcoral', width=1),
            opacity=0.7
        ))

        # Add horizontal line showing DMA total average
        #fig.add_hline(y=dma_total_avg, line_dash="solid", line_color="red", line_width=2,
        #              annotation_text=f"DMA Total Avg ({dma_total_avg:.4f} kg/m³)")

        # Add a horizontal line at zero for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="Zero Difference")

        # Update layout
        fig.update_layout(
            title=f"Difference Between DMA Total Average and {instrument} Rolling Average ({window_size}-point window)",
            xaxis_title="Time",
            yaxis_title=f"Density Difference (DMA total avg - {instrument} rolling avg) [kg/m³]",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot with instrument-specific filename
        filename = f"dma_totalavg_{instrument.lower()}_rolling_difference.html"
        fig.write_html(filename)
        print(f"DMA total avg - {instrument} rolling difference plot saved as {filename}")

        # Print statistics
        mean_diff = df_analysis[diff_column].mean()
        std_diff = df_analysis[diff_column].std()
        min_diff = df_analysis[diff_column].min()
        max_diff = df_analysis[diff_column].max()

        print(f"\nDMA Total Average - {instrument} Rolling Average Difference Statistics:")
        print(f"  Mean difference: {mean_diff:.6f} kg/m³")
        print(f"  Std deviation:   {std_diff:.6f} kg/m³")
        print(f"  Range: [{min_diff:.6f}, {max_diff:.6f}] kg/m³")

        # Check if there's a systematic bias
        if abs(mean_diff) > 2 * std_diff:
            print(
                f"  → Systematic bias detected: DMA total avg is consistently {'higher' if mean_diff > 0 else 'lower'} than {instrument} rolling avg")
        else:
            print(f"  → No significant systematic bias detected")

    except Exception as e:
        print(f"Error creating DMA total avg - {instrument} rolling difference plot: {e}")
##########################################################
def plot_three_densities_with_averages(df):
    """Plot all three density measurements with horizontal lines showing their total averages."""
    print("Creating three densities plot with averages...")

    try:
        # Calculate total averages for each density measurement
        dma_avg = df['Density'].mean()
        mgce1_avg = df['MGCE1_Density'].mean()
        dgfi1_avg = df['DGFI1_Density'].mean()

        print(f"DMA (Density) average: {dma_avg:.6f} kg/m³")
        print(f"MGCE1_Density average: {mgce1_avg:.6f} kg/m³")
        print(f"DGFI1_Density average: {dgfi1_avg:.6f} kg/m³")

        # Create the plot
        fig = go.Figure()

        # Add DMA density trace
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],  # Using index as x-axis (you can change to df['Timestamp'] if available)
            y=df['Density']-dma_avg,
            mode='lines',
            name='DMA Density - DMA average',
            line=dict(color='red', width=2)
        ))

        # Add MGCE1 density trace
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['MGCE1_Density']-mgce1_avg,
            mode='lines',
            name='MGCE1 Density - MGCE1 average',
            line=dict(color='blue', width=2)
        ))

        # Add DGFI1 density trace
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['DGFI1_Density']-dgfi1_avg,
            mode='lines',
            name='DGFI1 Density - DGFI1 average',
            line=dict(color='green', width=2)
        ))



        # Update layout
        fig.update_layout(
            title="Three Density Measurements with Total Averages",
            xaxis_title="Data Point Index",  # Change this if you have a time column
            yaxis_title="Density [kg/m³]",
            width=1200,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save the plot
        filename = "three_densities_with_averages.html"
        fig.write_html(filename)
        print(f"Three densities plot saved as {filename}")

        # Print comparative statistics
        print(f"\nComparative Statistics:")
        print(f"  DMA - MGCE1 difference: {dma_avg - mgce1_avg:.6f} kg/m³")
        print(f"  DMA - DGFI1 difference: {dma_avg - dgfi1_avg:.6f} kg/m³")
        print(f"  MGCE1 - DGFI1 difference: {mgce1_avg - dgfi1_avg:.6f} kg/m³")

        # Show which measurement has the highest/lowest average
        averages = {'DMA': dma_avg, 'MGCE1': mgce1_avg, 'DGFI1': dgfi1_avg}
        highest = max(averages, key=averages.get)
        lowest = min(averages, key=averages.get)

        print(f"  Highest average: {highest} ({averages[highest]:.6f} kg/m³)")
        print(f"  Lowest average: {lowest} ({averages[lowest]:.6f} kg/m³)")
        print(f"  Range: {averages[highest] - averages[lowest]:.6f} kg/m³")

        return fig

    except Exception as e:
        print(f"Error creating three densities plot: {e}")
        return None
###########################################################
def create_density_timeseries_plot_seperat(df, df_analysis, window_size=50):
    """Create two time series plots comparing each TrueDyne instrument with DMA on separate scales."""
    print("Creating density time series plots...")
    try:
        # Calculate rolling averages for all density measurements
        df_analysis['DMA_Density_Rolling'] = df_analysis['Density'].rolling(window=window_size, center=True).mean()
        df_analysis['MGCE1_Density_Rolling'] = df_analysis['MGCE1_Density'].rolling(window=window_size,
                                                                                    center=True).mean()
        df_analysis['DGFI1_Density_Rolling'] = df_analysis['DGFI1_Density'].rolling(window=window_size,
                                                                                    center=True).mean()

        # Calculate rolling standard deviations for error bars
        df_analysis['DMA_Density_Rolling_Std'] = df_analysis['DMA_Density_Rolling'].rolling(window=window_size, center=True).std()
        df_analysis['MGCE1_Density_Rolling_Std'] = df_analysis['MGCE1_Density_Rolling'].rolling(window=window_size,
                                                                                        center=True).std()
        df_analysis['DGFI1_Density_Rolling_Std'] = df_analysis['DGFI1_Density_Rolling'].rolling(window=window_size,
                                                                                        center=True).std()

        # Create error bar masks for every 100th point
        error_mask = np.arange(len(df_analysis)) % 100 == 0
        error_mask_2 = (np.arange(len(df_analysis)) + 50) % 100 == 0
        dma_error_array = np.where(error_mask_2, df_analysis['DMA_Density_Rolling_Std'], np.nan)
        mgce1_error_array = np.where(error_mask, df_analysis['MGCE1_Density_Rolling_Std'], np.nan)
        dgfi1_error_array = np.where(error_mask, df_analysis['DGFI1_Density_Rolling_Std'], np.nan)

        # PLOT 1: DGFI1 vs DMA
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add raw data lines (lighter/thinner) - DGFI1 on left axis
        fig1.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Density'],
            mode='lines',
            name='DGFI1 Density (raw)',
            line=dict(color='lightgreen', width=1),
            opacity=0.1
        ), secondary_y=False)

        # Add DMA raw data on right axis
        fig1.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Density'],
            mode='lines',
            name='DMA Density (raw)',
            line=dict(color='lightcoral', width=1),
            opacity=0.1
        ), secondary_y=True)

        # Add rolling average lines with error bars - DGFI1 on left axis
        fig1.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Density_Rolling'],
            error_y=dict(
                type='data',
                array=dgfi1_error_array,
                visible=True,
                color='black',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DGFI1 Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='green', width=3)
        ), secondary_y=False)

        # Add DMA rolling average on right axis
        fig1.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_Density_Rolling'],
            error_y=dict(
                type='data',
                array=dma_error_array,
                visible=True,
                color='black',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DMA Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='red', width=3)
        ), secondary_y=True)

        # Set y-axes titles for Plot 1
        fig1.update_yaxes(title_text="DGFI1 Density (kg/m³)", secondary_y=False)
        fig1.update_yaxes(title_text="DMA Density (kg/m³)", secondary_y=True)

        # Update layout for Plot 1
        fig1.update_layout(
            title=f"DGFI1 vs DMA Density Time Series with {window_size}-Point Rolling Averages",
            xaxis_title="Time",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save Plot 1
        fig1.write_html("density_timeseries_dgfi1_vs_dma.html")
        print("DGFI1 vs DMA density time series plot saved as density_timeseries_dgfi1_vs_dma.html")

        # PLOT 2: MGCE1 vs DMA
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add raw data lines (lighter/thinner) - MGCE1 on left axis
        fig2.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Density'],
            mode='lines',
            name='MGCE1 Density (raw)',
            line=dict(color='lightblue', width=1),
            opacity=0.1
        ), secondary_y=False)

        # Add DMA raw data on right axis
        fig2.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Density'],
            mode='lines',
            name='DMA Density (raw)',
            line=dict(color='lightcoral', width=1),
            opacity=0.1
        ), secondary_y=True)

        # Add rolling average lines with error bars - MGCE1 on left axis
        fig2.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Density_Rolling'],
            error_y=dict(
                type='data',
                array=mgce1_error_array,
                visible=True,
                color='black',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'MGCE1 Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)

        # Add DMA rolling average on right axis
        fig2.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_Density_Rolling'],
            error_y=dict(
                type='data',
                array=dma_error_array,
                visible=True,
                color='black',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DMA Density ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='red', width=3)
        ), secondary_y=True)

        # Set y-axes titles for Plot 2
        fig2.update_yaxes(title_text="MGCE1 Density (kg/m³)", secondary_y=False)
        fig2.update_yaxes(title_text="DMA Density (kg/m³)", secondary_y=True)

        # Update layout for Plot 2
        fig2.update_layout(
            title=f"MGCE1 vs DMA Density Time Series with {window_size}-Point Rolling Averages",
            xaxis_title="Time",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save Plot 2
        fig2.write_html("density_timeseries_mgce1_vs_dma.html")
        print("MGCE1 vs DMA density time series plot saved as density_timeseries_mgce1_vs_dma.html")

        # Print statistics
        print(f"\nDensity Statistics:")
        print(f"DMA Density:")
        print(f"  Mean: {df_analysis['Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['Density'].min():.6f}, {df_analysis['Density'].max():.6f}] kg/m³")

        print(f"\nMGCE1 Density:")
        print(f"  Mean: {df_analysis['MGCE1_Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['MGCE1_Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['MGCE1_Density'].min():.6f}, {df_analysis['MGCE1_Density'].max():.6f}] kg/m³")

        print(f"\nDGFI1 Density:")
        print(f"  Mean: {df_analysis['DGFI1_Density'].mean():.6f} kg/m³")
        print(f"  Std:  {df_analysis['DGFI1_Density'].std():.6f} kg/m³")
        print(f"  Range: [{df_analysis['DGFI1_Density'].min():.6f}, {df_analysis['DGFI1_Density'].max():.6f}] kg/m³")

    except Exception as e:
        print(f"Error creating density time series plots: {e}")
################################################################
def create_temperature_difference_plot(df, df_analysis, window_size=50):
    """Create plot showing temperature differences relative to Cell Temperature average with ambient temp."""
    print("Creating temperature difference plot...")
    try:
        # Calculate average of Cell temperature values (NEW BASELINE)
        cell_avg_temp = df_analysis['Cell_Temp'].mean()

        # Calculate differences from Cell temperature average
        df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'] = df_analysis['MGCE1_Temp'] - cell_avg_temp
        df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'] = df_analysis['DGFI1_Temp'] - cell_avg_temp
        df_analysis['Cell_Temp_Diff_from_Avg'] = df_analysis['Cell_Temp'] - cell_avg_temp

        # Calculate rolling averages and standard deviations
        df_analysis['MGCE1_Temp_Diff_Rolling'] = df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].rolling(
            window=window_size, center=True).mean()
        df_analysis['MGCE1_Temp_Diff_Rolling_Std'] = df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].rolling(
            window=window_size, center=True).std()

        df_analysis['DGFI1_Temp_Diff_Rolling'] = df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].rolling(
            window=window_size, center=True).mean()
        df_analysis['DGFI1_Temp_Diff_Rolling_Std'] = df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].rolling(
            window=window_size, center=True).std()

        df_analysis['Cell_Temp_Diff_Rolling'] = df_analysis['Cell_Temp_Diff_from_Avg'].rolling(window=window_size,
                                                                                               center=True).mean()
        df_analysis['Cell_Temp_Diff_Rolling_Std'] = df_analysis['Cell_Temp_Diff_from_Avg'].rolling(window=window_size,
                                                                                                   center=True).std()

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add raw difference lines (lighter/thinner) - LEFT Y-AXIS
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'],
            mode='lines',
            name='MGCE1 - Cell_avg (raw)',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'],
            mode='lines',
            name='DGFI1 - Cell_avg (raw)',
            line=dict(color='lightgreen', width=1),
            opacity=0.5
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Cell_Temp_Diff_from_Avg'],
            mode='lines',
            name='Cell - Cell_avg (raw)',
            line=dict(color='lightcoral', width=1),
            opacity=0.5
        ), secondary_y=False)

        # Create error bar masks for every 100th point
        error_mask = np.arange(len(df_analysis)) % 100 == 0
        mgce1_error_array = np.where(error_mask, df_analysis['MGCE1_Temp_Diff_Rolling_Std'], np.nan)
        dgfi1_error_array = np.where(error_mask, df_analysis['DGFI1_Temp_Diff_Rolling_Std'], np.nan)
        cell_error_array = np.where(error_mask, df_analysis['Cell_Temp_Diff_Rolling_Std'], np.nan)

        # Add rolling average lines with error bars (bolder) - LEFT Y-AXIS
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Temp_Diff_Rolling'],
            error_y=dict(
                type='data',
                array=mgce1_error_array,
                visible=True,
                color='blue',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'MGCE1 - Cell_avg ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Temp_Diff_Rolling'],
            error_y=dict(
                type='data',
                array=dgfi1_error_array,
                visible=True,
                color='green',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DGFI1 - Cell_avg ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='green', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Cell_Temp_Diff_Rolling'],
            error_y=dict(
                type='data',
                array=cell_error_array,
                visible=True,
                color='red',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'Cell - Cell_avg ({window_size}-pt avg ± 1σ every 100pts)',
            line=dict(color='red', width=3)
        ), secondary_y=False)

        # Add ambient temperature line - RIGHT Y-AXIS
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Env_Temp'],
            mode='lines',
            name='Outdoor Temperature, according to meteostat.net ',
            line=dict(color='orange', width=2, dash='dot'),
            opacity=0.8
        ), secondary_y=True)

        # Add a horizontal line at zero for reference on left axis
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="Cell Temperature Average")

        # Set y-axes titles
        fig.update_yaxes(title_text="Temperature Difference from Cell Average (°C)", secondary_y=False)
        fig.update_yaxes(title_text="Ambient Temperature (°C)", secondary_y=True)

        # Update layout
        fig.update_layout(
            title=f"Temperature Differences from Cell Temperature Average with Ambient Temperature Over Time (Cell avg = {cell_avg_temp:.3f}°C)",
            xaxis_title="Time",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot
        fig.write_html("temperature_differences_from_cell_avg.html")
        print("Temperature differences plot saved as temperature_differences_from_cell_avg.html")

        # Print statistics
        print(f"\nCell Average Temperature: {cell_avg_temp:.3f}°C")
        print(f"\nMGCE1 Differences from Cell average:")
        print(f"  Mean: {df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].mean():.3f}°C")
        print(f"  Std:  {df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].std():.3f}°C")
        print(
            f"  Range: [{df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].min():.3f}, {df_analysis['MGCE1_Temp_Diff_from_Cell_Avg'].max():.3f}]°C")

        print(f"\nDGFI1 Differences from Cell average:")
        print(f"  Mean: {df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].mean():.3f}°C")
        print(f"  Std:  {df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].std():.3f}°C")
        print(
            f"  Range: [{df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].min():.3f}, {df_analysis['DGFI1_Temp_Diff_from_Cell_Avg'].max():.3f}]°C")

        print(f"\nCell Temperature Differences from its own average:")
        print(f"  Mean: {df_analysis['Cell_Temp_Diff_from_Avg'].mean():.3f}°C")
        print(f"  Std:  {df_analysis['Cell_Temp_Diff_from_Avg'].std():.3f}°C")
        print(
            f"  Range: [{df_analysis['Cell_Temp_Diff_from_Avg'].min():.3f}, {df_analysis['Cell_Temp_Diff_from_Avg'].max():.3f}]°C")

        # Print ambient temperature stats if available
        if df_analysis['Env_Temp'].notna().any():
            print(f"\nAmbient Temperature:")
            print(f"  Mean: {df_analysis['Env_Temp'].mean():.1f}°C")
            print(f"  Range: [{df_analysis['Env_Temp'].min():.1f}, {df_analysis['Env_Temp'].max():.1f}]°C")

    except Exception as e:
        print(f"Error creating temperature difference plot: {e}")




################################################################
def create_pressure_difference_plot(df, df_analysis, window_size=50):
    """Create plot showing pressure differences relative to MGCE1 average."""
    print("Creating pressure difference plot...")
    try:
        # Calculate average of MGCE1 pressure values
        mgce1_avg_pressure = df_analysis['MGCE1_Pressure'].mean()

        # Calculate differences from MGCE1 average
        df_analysis['MGCE1_Press_Diff_from_Avg'] = df_analysis['MGCE1_Pressure'] - mgce1_avg_pressure
        df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'] = df_analysis['DGFI1_Pressure'] - mgce1_avg_pressure

        # Calculate rolling averages
        df_analysis['MGCE1_Press_Diff_Rolling'] = df_analysis['MGCE1_Press_Diff_from_Avg'].rolling(window=window_size,
                                                                                                   center=True).mean()
        df_analysis['MGCE1_Press_Diff_Rolling_Std'] = df_analysis['MGCE1_Press_Diff_from_Avg'].rolling(
            window=window_size, center=True).std()

        df_analysis['DGFI1_Press_Diff_Rolling'] = df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].rolling(
            window=window_size, center=True).mean()
        df_analysis['DGFI1_Press_Diff_Rolling_Std'] = df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].rolling(
            window=window_size, center=True).std()

        # Erstelle Masken für jeden 100. Datenpunkt
        error_mask = np.arange(len(df_analysis)) % 50 == 0

        # Erstelle Arrays mit NaN für Punkte ohne Error Bars
        mgce1_error_array = np.where(error_mask, df_analysis['MGCE1_Press_Diff_Rolling_Std'], np.nan)
        dgfi1_error_array = np.where(error_mask, df_analysis['DGFI1_Press_Diff_Rolling_Std'], np.nan)

        # Create the plot
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add raw difference lines (lighter/thinner)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Press_Diff_from_Avg'],
            mode='lines',
            name='MGCE1 - MGCE1_avg (raw)',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'],
            mode='lines',
            name='DGFI1 - MGCE1_avg (raw)',
            line=dict(color='lightgreen', width=1),
            opacity=0.5
        ), secondary_y=False)
        # Add rolling average lines with error bars (bolder)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['MGCE1_Press_Diff_Rolling'],
            error_y=dict(
                type='data',
                #array=df_analysis['MGCE1_Press_Diff_Rolling_Std'],
                array=mgce1_error_array,
                visible=True,
                color='blue',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'MGCE1 - MGCE1_avg ({window_size}-pt avg ± 1σ)',
            line=dict(color='blue', width=3)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DGFI1_Press_Diff_Rolling'],
            error_y=dict(
                type='data',
                #array=df_analysis['DGFI1_Press_Diff_Rolling_Std'],
                array=dgfi1_error_array,
                visible=True,
                color='green',
                thickness=1.5,
                width=3
            ),
            mode='lines',
            name=f'DGFI1 - MGCE1_avg ({window_size}-pt avg ± 1σ)',
            line=dict(color='green', width=3)
        ), secondary_y=False)



        # Add a horizontal line at zero for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="MGCE1 Average")

        # Add ambient pressure line - RIGHT Y-AXIS
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['Env_Pressure'],
            mode='lines',
            name='Ambient Pressure',
            line=dict(color='orange', width=2, dash='dot'),
            opacity=0.8
        ), secondary_y=True)

        # Set y-axes titles
        fig.update_yaxes(title_text="Pressure Difference from MGCE1 Average", secondary_y=False)
        fig.update_yaxes(title_text="Ambient Pressure (hPa)", secondary_y=True)

        # Update layout
        fig.update_layout(
            title=f"Pressure Differences from MGCE1 Average Over Time (MGCE1 avg = {mgce1_avg_pressure:.6f} bar)",
            xaxis_title="Time",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot
        fig.write_html("pressure_differences_from_mgce1_avg.html")
        print("Pressure differences plot saved as pressure_differences_from_mgce1_avg.html")

        # Print statistics
        print(f"\nMGCE1 Average Pressure: {mgce1_avg_pressure:.6f}")
        print(f"\nMGCE1 Differences from its own average:")
        print(f"  Mean: {df_analysis['MGCE1_Press_Diff_from_Avg'].mean():.6f}")
        print(f"  Std:  {df_analysis['MGCE1_Press_Diff_from_Avg'].std():.6f}")
        print(
            f"  Range: [{df_analysis['MGCE1_Press_Diff_from_Avg'].min():.6f}, {df_analysis['MGCE1_Press_Diff_from_Avg'].max():.6f}]")

        print(f"\nDGFI1 Differences from MGCE1 average:")
        print(f"  Mean: {df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].mean():.6f}")
        print(f"  Std:  {df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].std():.6f}")
        print(
            f"  Range: [{df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].min():.6f}, {df_analysis['DGFI1_Press_Diff_from_MGCE1_Avg'].max():.6f}]")

    except Exception as e:
        print(f"Error creating pressure difference plot: {e}")
###################################################################
def create_dma_truedyne_difference_plot(df, df_analysis, window_size=50):
    """Create time series plot of DMA vs both TrueDyne instruments difference with rolling average."""
    print("Creating DMA vs TrueDyne difference plot...")
    try:
        # Calculate differences: DMA - each TrueDyne
        df_analysis['DMA_MGCE1_Diff'] = df_analysis['Density'] - df_analysis['MGCE1_Density']
        df_analysis['DMA_DGFI1_Diff'] = df_analysis['Density'] - df_analysis['DGFI1_Density']

        # Calculate rolling averages
        df_analysis['DMA_MGCE1_Diff_Rolling'] = df_analysis['DMA_MGCE1_Diff'].rolling(window=window_size,
                                                                                      center=True).mean()
        df_analysis['DMA_DGFI1_Diff_Rolling'] = df_analysis['DMA_DGFI1_Diff'].rolling(window=window_size,
                                                                                      center=True).mean()

        # Create the plot
        fig = go.Figure()

        # Add raw difference lines (lighter/thinner)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_MGCE1_Diff'],
            mode='lines',
            name='DMA - MGCE1 (raw)',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_DGFI1_Diff'],
            mode='lines',
            name='DMA - DGFI1 (raw)',
            line=dict(color='lightcoral', width=1),
            opacity=0.5
        ))

        # Add rolling average lines (bolder)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_MGCE1_Diff_Rolling'],
            mode='lines',
            name=f'DMA - MGCE1 ({window_size}-pt avg)',
            line=dict(color='blue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_DGFI1_Diff_Rolling'],
            mode='lines',
            name=f'DMA - DGFI1 ({window_size}-pt avg)',
            line=dict(color='red', width=3)
        ))

        # Add a horizontal line at zero for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="Measurment Value of the DMA")

        # Update layout
        fig.update_layout(
            title=f"DMA Density vs TrueDyne Instruments Differences Over Time (with {window_size}-point rolling average)",
            xaxis_title="Time",
            yaxis_title="Density Difference (kg/m³)",
            width=1200,
            height=600,
            showlegend=True
        )

        # Save the plot
        fig.write_html("dma_truedyne_differences.html")
        print("DMA vs TrueDyne differences plot saved as dma_truedyne_differences.html")

    except Exception as e:
        print(f"Error creating DMA vs TrueDyne difference plot: {e}")
################################################################
def create_time_series_analysis(df, df_analysis):
    """Create comprehensive time series analysis plots."""
    print("Creating time series analysis...")
    try:
        fig_time = make_subplots(
            rows=3, cols=2,
            subplot_titles=('MGCE1 vs DGFI1 Density', 'MGCE1 vs DGFI1 Pressure',
                            'MGCE1 vs DGFI1 Temperature', 'Cell Temperature vs Time',
                            'Density Comparison', 'Pressure Comparison'),
            vertical_spacing=0.08
        )

        # Density comparison over time
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['MGCE1_Density'],
                       name='MGCE1 Density', line=dict(color='blue')),
            row=1, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['DGFI1_Density'],
                       name='DGFI1 Density', line=dict(color='red')),
            row=1, col=1
        )

        # Pressure comparison over time
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['MGCE1_Pressure'],
                       name='MGCE1 Pressure', line=dict(color='green')),
            row=1, col=2
        )
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['DGFI1_Pressure'],
                       name='DGFI1 Pressure', line=dict(color='orange')),
            row=1, col=2
        )

        # Temperature comparison over time
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['MGCE1_Temp'],
                       name='MGCE1 Temp', line=dict(color='purple')),
            row=2, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['DGFI1_Temp'],
                       name='DGFI1 Temp', line=dict(color='brown')),
            row=2, col=1
        )

        # Cell temperature over time
        fig_time.add_trace(
            go.Scatter(x=df['Timestamp'], y=df_analysis['Cell_Temp'],
                       name='Cell Temp', line=dict(color='black')),
            row=2, col=2
        )

        # Density vs Cell Temperature
        fig_time.add_trace(
            go.Scatter(x=df_analysis['Cell_Temp'], y=df_analysis['MGCE1_Density'],
                       mode='markers', name='MGCE1 vs Cell Temp',
                       marker=dict(color='blue', size=3, opacity=0.6)),
            row=3, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=df_analysis['Cell_Temp'], y=df_analysis['DGFI1_Density'],
                       mode='markers', name='DGFI1 vs Cell Temp',
                       marker=dict(color='red', size=3, opacity=0.6)),
            row=3, col=1
        )

        # MGCE1 vs DGFI1 Direct Comparison
        fig_time.add_trace(
            go.Scatter(x=df_analysis['MGCE1_Density'], y=df_analysis['DGFI1_Density'],
                       mode='markers', name='MGCE1 vs DGFI1 Density',
                       marker=dict(color='green', size=3, opacity=0.6)),
            row=3, col=2
        )

        fig_time.update_layout(height=1200, title_text="Comprehensive Time Series and Correlation Analysis")
        fig_time.write_html("time_series_analysis.html")
        print("Time series analysis saved as time_series_analysis.html")
    except Exception as e:
        print(f"Error creating time series analysis: {e}")


def perform_statistical_analysis(correlation_matrix, numerical_cols):
    """Perform detailed statistical correlation analysis."""
    print("\n" + "=" * 60)
    print("DETAILED CORRELATION ANALYSIS")
    print("=" * 60)

    # Find strongest correlations
    corr_pairs = []
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            var1, var2 = numerical_cols[i], numerical_cols[j]
            corr_val = correlation_matrix.loc[var1, var2]
            corr_pairs.append((var1, var2, corr_val))

    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nStrongest Correlations (|r| > 0.3):")
    print("-" * 50)
    for var1, var2, corr in corr_pairs:
        if abs(corr) > 0.3:
            significance = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*"
            print(f"{var1:20} vs {var2:20}: {corr:6.3f} {significance}")


def perform_instrument_comparison(df_analysis):
    """Analyze differences between MGCE1 and DGFI1 instruments."""
    print("\n" + "=" * 60)
    print("INSTRUMENT COMPARISON (MGCE1 vs DGFI1)")
    print("=" * 60)

    # Calculate differences between instruments
    df_analysis['Density_Diff'] = df_analysis['MGCE1_Density'] - df_analysis['DGFI1_Density']
    df_analysis['Pressure_Diff'] = df_analysis['MGCE1_Pressure'] - df_analysis['DGFI1_Pressure']
    df_analysis['Temp_Diff'] = df_analysis['MGCE1_Temp'] - df_analysis['DGFI1_Temp']

    print(f"Density Difference (MGCE1 - DGFI1):")
    print(f"  Mean: {df_analysis['Density_Diff'].mean():.6f}")
    print(f"  Std:  {df_analysis['Density_Diff'].std():.6f}")
    print(f"  Range: [{df_analysis['Density_Diff'].min():.6f}, {df_analysis['Density_Diff'].max():.6f}]")

    print(f"\nPressure Difference (MGCE1 - DGFI1):")
    print(f"  Mean: {df_analysis['Pressure_Diff'].mean():.6f}")
    print(f"  Std:  {df_analysis['Pressure_Diff'].std():.6f}")
    print(f"  Range: [{df_analysis['Pressure_Diff'].min():.6f}, {df_analysis['Pressure_Diff'].max():.6f}]")

    print(f"\nTemperature Difference (MGCE1 - DGFI1):")
    print(f"  Mean: {df_analysis['Temp_Diff'].mean():.6f}")
    print(f"  Std:  {df_analysis['Temp_Diff'].std():.6f}")
    print(f"  Range: [{df_analysis['Temp_Diff'].min():.6f}, {df_analysis['Temp_Diff'].max():.6f}]")


def create_advanced_correlation_plots(df_analysis):
    """Create advanced correlation plots with trend lines."""
    print("Creating advanced correlation plots...")
    try:
        fig_advanced = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MGCE1 vs DGFI1 Density', 'MGCE1 vs DGFI1 Pressure',
                            'MGCE1 vs DGFI1 Temperature', 'Density vs Cell Temperature'),
            vertical_spacing=0.1
        )

        # MGCE1 vs DGFI1 Density with trend line
        fig_advanced.add_trace(
            go.Scatter(x=df_analysis['MGCE1_Density'], y=df_analysis['DGFI1_Density'],
                       mode='markers', name='Density Correlation',
                       marker=dict(size=4, opacity=0.6, color='blue')),
            row=1, col=1
        )

        # Add trend line for density
        z = np.polyfit(df_analysis['MGCE1_Density'], df_analysis['DGFI1_Density'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_analysis['MGCE1_Density'].min(), df_analysis['MGCE1_Density'].max(), 100)
        fig_advanced.add_trace(
            go.Scatter(x=x_trend, y=p(x_trend), mode='lines', name='Density Trend',
                       line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Similar for pressure and temperature...
        fig_advanced.add_trace(
            go.Scatter(x=df_analysis['MGCE1_Pressure'], y=df_analysis['DGFI1_Pressure'],
                       mode='markers', name='Pressure Correlation',
                       marker=dict(size=4, opacity=0.6, color='green')),
            row=1, col=2
        )

        fig_advanced.add_trace(
            go.Scatter(x=df_analysis['MGCE1_Temp'], y=df_analysis['DGFI1_Temp'],
                       mode='markers', name='Temperature Correlation',
                       marker=dict(size=4, opacity=0.6, color='orange')),
            row=2, col=1
        )

        fig_advanced.add_trace(
            go.Scatter(x=df_analysis['Cell_Temp'], y=df_analysis['MGCE1_Density'],
                       mode='markers', name='MGCE1 Density vs Cell Temp',
                       marker=dict(size=4, opacity=0.6, color='purple')),
            row=2, col=2
        )

        fig_advanced.update_layout(height=800, title_text="Advanced Correlation Analysis with Trend Lines")
        fig_advanced.write_html("advanced_correlation_plots.html")
        print("Advanced correlation plots saved as advanced_correlation_plots.html")
    except Exception as e:
        print(f"Error creating advanced correlation plots: {e}")




#############################################

#################################################################################
def print_summary_recommendations():
    """Print summary and recommendations."""
    print("\n" + "=" * 60)
    print("SUMMARY RECOMMENDATIONS")
    print("=" * 60)
    print("1. Check the strongest correlations identified above")
    print("2. Investigate any unexpected weak correlations between related measurements")
    print("3. Consider the systematic differences between MGCE1 and DGFI1 instruments")
    print("4. Look for time-dependent trends in the data")
    print("5. Examine outliers in the scatter plots for data quality issues")


def main():
    """
    Main function to perform correlation analysis on measurement data.
    """
    print("Starting correlation analysis...")

    # Load and prepare data
    print("Loading data...")
    df, df_analysis, numerical_cols = load_and_prepare_data()
    #create_density_timeseries_plot(df, df_analysis, window_size=50)
    create_density_timeseries_plot_seperat(df, df_analysis, window_size=50)

    # Create both difference plots
    create_dma_truedyne_rolling_difference_plot(df, df_analysis, instrument='MGCE1', window_size=50)
    create_dma_truedyne_rolling_difference_plot(df, df_analysis, instrument='DGFI1', window_size=50)
    #create_temperature_difference_plot(df, df_analysis)
    # create_pressure_difference_plot(df, df_analysis, window_size=50)
    # # Print data overview
    # print_data_overview(df, df_analysis)
    # create_dma_truedyne_difference_plot(df, df_analysis, window_size=70)  # Adjust window_size as needed
    #
    #
    # # Calculate correlation matrix
    # print("Calculating correlation matrix...")
    # correlation_matrix = df_analysis[numerical_cols].corr()
    # print("Correlation matrix calculated!")
    #
    # # Generate visualizations and save as HTML files
    # print("\nGenerating visualizations (saving as HTML files)...")
    # create_correlation_heatmap(correlation_matrix)
    # create_scatter_matrix(df_analysis)
    # create_time_series_analysis(df, df_analysis)
    # create_advanced_correlation_plots(df_analysis)
    #
    # print("\nPerforming statistical analysis...")
    # perform_statistical_analysis(correlation_matrix, numerical_cols)
    #
    # print("\nPerforming instrument comparison...")
    # perform_instrument_comparison(df_analysis)
    # #
    # print_summary_recommendations()
    # #
    # print("\n" + "=" * 60)
    # print("ANALYSIS COMPLETE!")
    # print("=" * 60)
    # print("HTML files generated:")
    # print("- correlation_heatmap.html")
    # print("- scatter_matrix.html")
    # print("- time_series_analysis.html")
    # print("- advanced_correlation_plots.html")
    # print("\nOpen these files in your web browser to view the interactive plots.")
    # print("=" * 60)


if __name__ == "__main__":
    main()