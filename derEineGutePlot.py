import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prepare_density_data(df, instrument, dma_mean=None):
    """
    Prepare density data for plotting by applying offset correction and centering.

    Args:
        df: DataFrame with fused data
        instrument: Instrument name ('MGCE1' or 'DGFI1')
        dma_mean: DMA mean for centering (if None, will be calculated)

    Returns:
        dict: Contains prepared data arrays and statistics
    """
    stats = calculate_unexplainable_difference(df, instrument)

    if stats is None:
        return None

    # Get raw data
    dma_reference = df['Density']
    truedyne_test = df[f'TrueDyne {instrument} Density (Average)']
    time_data = pd.to_datetime(df['Timestamp'])

    # Remove NaN values
    valid_mask = ~(pd.isna(dma_reference) | pd.isna(truedyne_test))
    dma_clean = dma_reference[valid_mask]
    truedyne_clean = truedyne_test[valid_mask]
    time_clean = time_data[valid_mask]

    # Calculate DMA mean if not provided
    if dma_mean is None:
        dma_mean = np.mean(dma_clean)

    # Apply offset correction to TrueDyne
    truedyne_corrected = truedyne_clean - stats['systematic_offset_kg_m3']

    # Center both around DMA mean
    dma_centered = dma_clean - dma_mean
    truedyne_centered = truedyne_corrected - dma_mean

    return {
        'time': time_clean,
        'dma_centered': dma_centered,
        'truedyne_centered': truedyne_centered,
        'dma_mean': dma_mean,
        'stats': stats
    }


def prepare_scale_data(df):
    """
    Prepare scale weight data for plotting by centering around its own mean.

    Args:
        df: DataFrame with fused data

    Returns:
        dict: Contains prepared scale data
    """
    scale_weight = df['Scale Weight (Average) [g]']
    time_data = pd.to_datetime(df['Timestamp'])

    # Remove NaN values
    valid_mask = ~pd.isna(scale_weight)
    scale_clean = scale_weight[valid_mask]
    time_clean = time_data[valid_mask]

    # Center around scale mean
    scale_mean = np.mean(scale_clean)
    scale_centered = scale_clean - scale_mean

    return {
        'time': time_clean,
        'scale_centered': scale_centered,
        'scale_mean': scale_mean
    }


def calculate_rolling_averages(data_dict, rolling_window=50):
    """
    Calculate rolling averages for all data series.

    Args:
        data_dict: Dictionary containing time series data
        rolling_window: Window size for rolling average

    Returns:
        dict: Updated dictionary with rolling averages added
    """
    result = data_dict.copy()

    # Calculate rolling averages for density data
    if 'dma_centered' in data_dict:
        result['dma_rolling'] = pd.Series(data_dict['dma_centered']).rolling(
            window=rolling_window, center=True).mean()

    if 'truedyne_centered' in data_dict:
        result['truedyne_rolling'] = pd.Series(data_dict['truedyne_centered']).rolling(
            window=rolling_window, center=True).mean()

    # Calculate rolling average for scale data
    if 'scale_centered' in data_dict:
        result['scale_rolling'] = pd.Series(data_dict['scale_centered']).rolling(
            window=rolling_window, center=True).mean()

    return result


def calculate_correlations(df, density_data, scale_data):
    """
    Calculate correlation coefficients between different measurements.

    Args:
        df: Original DataFrame
        density_data: Dictionary with density data
        scale_data: Dictionary with scale data

    Returns:
        dict: Correlation coefficients
    """
    correlations = {}

    # Get original data for correlation calculation
    dma_reference = df['Density']
    scale_weight = df['Scale Weight (Average) [g]']
    mgce1_density = df['TrueDyne MGCE1 Density (Average)']
    time_data = pd.to_datetime(df['Timestamp'])

    # DMA-Scale correlation
    common_times = pd.Index(density_data['time']).intersection(pd.Index(scale_data['time']))
    if len(common_times) > 1:
        dma_for_corr = dma_reference[time_data.isin(common_times)]
        scale_for_corr = scale_weight[time_data.isin(common_times)]
        valid_mask = ~(pd.isna(dma_for_corr) | pd.isna(scale_for_corr))

        if np.sum(valid_mask) > 1:
            correlations['dma_scale'] = np.corrcoef(
                dma_for_corr[valid_mask], scale_for_corr[valid_mask])[0, 1]
            correlations['dma_scale_points'] = np.sum(valid_mask)
        else:
            correlations['dma_scale'] = np.nan
            correlations['dma_scale_points'] = 0
    else:
        correlations['dma_scale'] = np.nan
        correlations['dma_scale_points'] = 0

    # MGCE1-Scale correlation
    if len(common_times) > 1:
        mgce1_for_corr = mgce1_density[time_data.isin(common_times)]
        valid_mask = ~(pd.isna(mgce1_for_corr) | pd.isna(scale_for_corr))

        if np.sum(valid_mask) > 1:
            correlations['mgce1_scale'] = np.corrcoef(
                mgce1_for_corr[valid_mask], scale_for_corr[valid_mask])[0, 1]
            correlations['mgce1_scale_points'] = np.sum(valid_mask)
        else:
            correlations['mgce1_scale'] = np.nan
            correlations['mgce1_scale_points'] = 0
    else:
        correlations['mgce1_scale'] = np.nan
        correlations['mgce1_scale_points'] = 0

    return correlations


def add_density_traces(fig, data_dict, instrument, rolling_window, show_raw=True, show_rolling=True):
    """
    Add density traces to the plotly figure.

    Args:
        fig: Plotly figure object
        data_dict: Dictionary with prepared data
        instrument: Instrument name
        rolling_window: Rolling window size for legend
        show_raw: Whether to show raw data traces
        show_rolling: Whether to show rolling average traces
    """
    # DMA traces
    if show_raw:
        fig.add_trace(go.Scatter(
            x=data_dict['time'],
            y=data_dict['dma_centered'],
            mode='lines',
            name='DMA Reference (centered)',
            line=dict(color='red', width=1, dash='dot'),
            opacity=0.5,
            visible=True
        ), secondary_y=False)

    if show_rolling:
        fig.add_trace(go.Scatter(
            x=data_dict['time'],
            y=data_dict['dma_rolling'],
            mode='lines',
            name=f'DMA Rolling Avg ({rolling_window} pts)',
            line=dict(color='red', width=2),
            visible=True
        ), secondary_y=False)

    # TrueDyne traces
    if show_raw:
        fig.add_trace(go.Scatter(
            x=data_dict['time'],
            y=data_dict['truedyne_centered'],
            mode='lines',
            name=f'{instrument} Calibrated (centered)',
            line=dict(color='blue', width=1, dash='dot'),
            opacity=0.5,
            visible=True
        ), secondary_y=False)

    if show_rolling:
        fig.add_trace(go.Scatter(
            x=data_dict['time'],
            y=data_dict['truedyne_rolling'],
            mode='lines',
            name=f'{instrument} Rolling Avg ({rolling_window} pts)',
            line=dict(color='blue', width=2),
            visible=True
        ), secondary_y=False)


def add_scale_traces(fig, scale_data, rolling_window, show_raw=True, show_rolling=True):
    """
    Add scale weight traces to the plotly figure.

    Args:
        fig: Plotly figure object
        scale_data: Dictionary with prepared scale data
        rolling_window: Rolling window size for legend
        show_raw: Whether to show raw data traces
        show_rolling: Whether to show rolling average traces
    """
    if show_raw:
        fig.add_trace(go.Scatter(
            x=scale_data['time'],
            y=scale_data['scale_centered'],
            mode='lines',
            name='Scale Weight (centered)',
            line=dict(color='orange', width=1, dash='dot'),
            opacity=0.5,
            visible=True,
            yaxis='y2'
        ), secondary_y=True)

    if show_rolling:
        fig.add_trace(go.Scatter(
            x=scale_data['time'],
            y=scale_data['scale_rolling'],
            mode='lines',
            name=f'Scale Weight Rolling Avg ({rolling_window} pts)',
            line=dict(color='orange', width=2),
            visible=True,
            yaxis='y2'
        ), secondary_y=True)


def create_statistics_annotation(density_data_list, scale_data, correlations, instruments):
    """
    Create the statistics text annotation for the plot.

    Args:
        density_data_list: List of density data dictionaries
        scale_data: Scale data dictionary
        correlations: Correlation coefficients dictionary
        instruments: List of instrument names

    Returns:
        str: Formatted statistics text
    """
    stats_text = f"DMA Mean: {density_data_list[0]['dma_mean']:.3f} kg/m³<br>"
    stats_text += f"Scale Mean: {scale_data['scale_mean']:.3f} g<br><br>"

    # Add statistics for each instrument
    for i, (data_dict, instrument) in enumerate(zip(density_data_list, instruments)):
        stats = data_dict['stats']
        bias_direction = "higher" if stats['systematic_offset_kg_m3'] > 0 else "lower"

        if i > 0:
            stats_text += "<br>"

        stats_text += (
            f"{instrument} bias: {stats['systematic_offset_kg_m3']:.3f} kg/m³ ({bias_direction})<br>"
            f"Residual RMS: {stats['residual_rms_kg_m3']:.3f} kg/m³<br>"
            f"Correlation: {stats['correlation_coefficient']:.4f}<br>"
        )

    # Add correlation information
    stats_text += f"<br>DMA-Scale Correlation: {correlations['dma_scale']:.4f}<br>"
    stats_text += f"MGCE1-Scale Correlation: {correlations['mgce1_scale']:.4f}"

    return stats_text


def plot_offset_corrected_comparison(df, instrument='MGCE1', second_instrument=None,
                                     rolling_window=50, show_raw_density=True,
                                     show_rolling_density=True, show_raw_scale=True,
                                     show_rolling_scale=True):
    """
    Create a plot showing the comparison after offset correction and mean subtraction.
    DMA is the reference, TrueDyne is corrected to match DMA.
    Both density and scale values are centered around their respective means.
    Scale weight is plotted on a separate right y-axis.

    Args:
        df: DataFrame with fused data
        instrument: Primary instrument ('MGCE1' or 'DGFI1')
        second_instrument: Optional second instrument to add to the same plot
        rolling_window: Window size for rolling average (default: 50)
        show_raw_density: Whether to show raw density traces
        show_rolling_density: Whether to show rolling average density traces
        show_raw_scale: Whether to show raw scale traces
        show_rolling_scale: Whether to show rolling average scale traces
    """




    # Prepare data for primary instrument
    density_data = prepare_density_data(df, instrument)
    if density_data is None:
        print(f"No valid data for {instrument}")
        return

    # Add rolling averages
    density_data = calculate_rolling_averages(density_data, rolling_window)

    # Prepare data for second instrument if specified
    density_data_list = [density_data]
    instruments = [instrument]

    if second_instrument:
        density_data2 = prepare_density_data(df, second_instrument, density_data['dma_mean'])
        if density_data2 is not None:
            density_data2 = calculate_rolling_averages(density_data2, rolling_window)
            density_data_list.append(density_data2)
            instruments.append(second_instrument)

    # Prepare scale data (centered around its own mean)
    scale_data = prepare_scale_data(df)
    scale_data = calculate_rolling_averages(scale_data, rolling_window)

    # Calculate correlations
    correlations = calculate_correlations(df, density_data, scale_data)

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add primary instrument traces
    add_density_traces(fig, density_data, instrument, rolling_window,
                       show_raw_density, show_rolling_density)

    # Add second instrument traces if available
    if second_instrument and len(density_data_list) > 1:
        density_data2 = density_data_list[1]

        if show_raw_density:
            fig.add_trace(go.Scatter(
                x=density_data2['time'],
                y=density_data2['truedyne_centered'],
                mode='lines',
                name=f'{second_instrument} Calibrated (centered)',
                line=dict(color='green', width=1, dash='dot'),
                opacity=0.5,
                visible=True
            ), secondary_y=False)

        if show_rolling_density:
            fig.add_trace(go.Scatter(
                x=density_data2['time'],
                y=density_data2['truedyne_rolling'],
                mode='lines',
                name=f'{second_instrument} Rolling Avg ({rolling_window} pts)',
                line=dict(color='green', width=2),
                visible=True
            ), secondary_y=False)

    # Add scale traces
    add_scale_traces(fig, scale_data, rolling_window, show_raw_scale, show_rolling_scale)

    # Add zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7)

    # Set y-axes titles
    fig.update_yaxes(title_text="Density Deviation from DMA Mean [kg/m³]", secondary_y=False)
    fig.update_yaxes(title_text="Scale Weight Deviation from Mean [g]", secondary_y=True)

    # Update layout
    title_text = f'TrueDyne {instrument} Calibrated to DMA Reference (Means Subtracted)'
    if second_instrument:
        title_text = f'TrueDyne {instrument} & {second_instrument} Calibrated to DMA Reference (Means Subtracted)'

    fig.update_layout(
        title=title_text,
        height=600,
        template='plotly_white',
        xaxis_title="Time"
    )

    # Add statistics annotation
    stats_text = create_statistics_annotation(density_data_list, scale_data, correlations, instruments)

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

    # Save plot
    filename = f"truedyne_{instrument.lower()}_calibration_to_dma_centered_with_scale.html"
    if second_instrument:
        filename = f"truedyne_{instrument.lower()}_{second_instrument.lower()}_calibration_to_dma_centered_with_scale.html"

    fig.write_html(filename)
    print(f"Saved: {filename}")
    print(f"DMA-Scale Weight Correlation: {correlations['dma_scale']:.4f}")
    print(f"Correlation calculated using {correlations['dma_scale_points']} paired data points")
    print(f"MGCE1-Scale correlation calculated using {correlations['mgce1_scale_points']} paired data points")

    return fig







def calculate_unexplainable_difference(df, instrument='MGCE1', density_column='Density'):
    """
    Calculate the unexplainable difference between DMA (reference) and TrueDyne (unit under test)
    after removing systematic offset from TrueDyne.
    This shows the remaining noise/time instability after perfect calibration of TrueDyne.

    Args:
        df: DataFrame with fused data
        instrument: 'MGCE1' or 'DGFI1'
        density_column: Column name for DMA density data (reference)

    Returns:
        dict: Statistics about unexplainable differences
    """



    # DMA is the reference, TrueDyne is unit under test
    dma_reference = df[density_column]
    truedyne_test = df[f'TrueDyne {instrument} Density (Average)']

    # Remove rows with NaN values
    valid_mask = ~(pd.isna(dma_reference) | pd.isna(truedyne_test))
    dma_clean = dma_reference[valid_mask]
    truedyne_clean = truedyne_test[valid_mask]

    if len(dma_clean) == 0:
        return None

    # Calculate the systematic offset (TrueDyne bias relative to DMA reference)
    systematic_offset = np.mean(truedyne_clean - dma_clean)

    # Remove the systematic offset from TrueDyne (calibrate TrueDyne to DMA)
    truedyne_corrected = truedyne_clean - systematic_offset

    # Calculate residual differences after calibrating TrueDyne to DMA
    residual_differences = truedyne_corrected - dma_clean

    # Calculate statistics (all in kg/m³)
    stats = {
        'systematic_offset_kg_m3': systematic_offset,  # Positive = TrueDyne reads high
        'residual_std_kg_m3': np.std(residual_differences),
        'residual_mean_kg_m3': np.mean(residual_differences),  # Should be ~0 after correction
        'residual_max_abs_kg_m3': np.max(np.abs(residual_differences)),
        'residual_rms_kg_m3': np.sqrt(np.mean(residual_differences ** 2)),
        'correlation_coefficient': np.corrcoef(truedyne_corrected, dma_clean)[0, 1],
        'number_of_points': len(residual_differences)
    }

    return stats