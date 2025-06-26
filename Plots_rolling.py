import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys


def load_and_parse_data(filename):
    """Load and parse the fused CSV data"""
    try:
        df = pd.read_csv(filename)
        df['parsed_timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
        df['local_time'] = df['parsed_timestamp'].dt.tz_convert('Europe/Zurich')
        df['DMA_Density_kg_m3'] = df['Density'] * 1000

        print(f"Loaded {len(df)} data points")
        print(f"Time range: {df['local_time'].min()} to {df['local_time'].max()}")
        print(f"DMA density converted from g/cm³ to kg/m³")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_overview_plot(df, output_prefix="time_series", rolling_window=50):
    """Create overview plot with all measurements"""
    has_temp = 'TrueDyne MGCE1 Temp (Average)' in df.columns

    # Always use 3 subplots: Density, Pressure, Temperature
    rows = 3
    titles = ('Density Measurements', 'Pressure Measurements', 'Temperature Measurements')
    height = 900

    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.08)
    time_data = df['local_time']

    # Calculate rolling averages and uncertainties
    dma_density_rolling = df['DMA_Density_kg_m3'].rolling(window=rolling_window, center=True).mean()

    mgce1_density_rolling = df['TrueDyne MGCE1 Density (Average)'].rolling(window=rolling_window, center=True).mean()
    mgce1_density_uncertainty_rolling = (df['TrueDyne MGCE1 Density Uncertainty (std)'] ** 2).rolling(
        window=rolling_window, center=True).mean().apply(lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dgfi1_density_rolling = df['TrueDyne DGFI1 Density (Average)'].rolling(window=rolling_window, center=True).mean()
    dgfi1_density_uncertainty_rolling = (df['TrueDyne DGFI1 Density Uncertainty (std)'] ** 2).rolling(
        window=rolling_window, center=True).mean().apply(lambda x: x ** 0.5 / (rolling_window ** 0.5))

    mgce1_press_rolling = df['TrueDyne MGCE1 Press (Average)'].rolling(window=rolling_window, center=True).mean()
    mgce1_press_uncertainty_rolling = (df['TrueDyne MGCE1 Press Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                                  center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dgfi1_press_rolling = df['TrueDyne DGFI1 Press (Average)'].rolling(window=rolling_window, center=True).mean()
    dgfi1_press_uncertainty_rolling = (df['TrueDyne DGFI1 Press Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                                  center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dma_temp_rolling = df['T(cell)'].rolling(window=rolling_window, center=True).mean()

    # Density plot
    # Original data (no error bars)
    fig.add_trace(go.Scatter(
        x=time_data, y=df['DMA_Density_kg_m3'],
        mode='lines+markers', name='DMA Density',
        line=dict(color='red', width=1), marker=dict(size=3),
        opacity=0.5
    ), row=1, col=1)

    # Rolling average with error bars
    fig.add_trace(go.Scatter(
        x=time_data, y=dma_density_rolling,
        mode='lines', name='DMA Density (Rolling Avg)',
        line=dict(color='red', width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Density (Average)'],
        mode='lines+markers', name='MGCE1 Density',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_density_rolling,
        error_y=dict(type='data', array=mgce1_density_uncertainty_rolling, visible=True),
        mode='lines', name='MGCE1 Density (Rolling Avg)',
        line=dict(color='green', width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Density (Average)'],
        mode='lines+markers', name='DGFI1 Density',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_density_rolling,
        error_y=dict(type='data', array=dgfi1_density_uncertainty_rolling, visible=True),
        mode='lines', name='DGFI1 Density (Rolling Avg)',
        line=dict(color='blue', width=3)
    ), row=1, col=1)

    # Pressure plot
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press (Average)'],
        mode='lines+markers', name='MGCE1 Pressure',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_press_rolling,
        error_y=dict(type='data', array=mgce1_press_uncertainty_rolling, visible=True),
        mode='lines', name='MGCE1 Pressure (Rolling Avg)',
        line=dict(color='green', width=3)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press (Average)'],
        mode='lines+markers', name='DGFI1 Pressure',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_press_rolling,
        error_y=dict(type='data', array=dgfi1_press_uncertainty_rolling, visible=True),
        mode='lines', name='DGFI1 Pressure (Rolling Avg)',
        line=dict(color='blue', width=3)
    ), row=2, col=1)

    # Temperature plot - ALL temperatures in one subplot
    # DMA Cell Temperature
    fig.add_trace(go.Scatter(
        x=time_data, y=df['T(cell)'],
        mode='lines+markers', name='DMA Cell Temperature',
        line=dict(color='red', width=1), marker=dict(size=2),
        opacity=0.5
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=dma_temp_rolling,
        mode='lines', name='DMA Cell Temp (Rolling Avg)',
        line=dict(color='red', width=3)
    ), row=3, col=1)

    # TrueDyne temperatures (if available)
    if has_temp:
        mgce1_temp_rolling = df['TrueDyne MGCE1 Temp (Average)'].rolling(window=rolling_window, center=True).mean()
        mgce1_temp_uncertainty_rolling = (df['TrueDyne MGCE1 Temp Uncertainty (std)'] ** 2).rolling(
            window=rolling_window, center=True).mean().apply(lambda x: x ** 0.5 / (rolling_window ** 0.5))

        dgfi1_temp_rolling = df['TrueDyne DGFI1 Temp (Average)'].rolling(window=rolling_window, center=True).mean()
        dgfi1_temp_uncertainty_rolling = (df['TrueDyne DGFI1 Temp Uncertainty (std)'] ** 2).rolling(
            window=rolling_window, center=True).mean().apply(lambda x: x ** 0.5 / (rolling_window ** 0.5))

        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne MGCE1 Temp (Average)'],
            mode='lines+markers', name='MGCE1 Temperature',
            line=dict(color='green', width=1), marker=dict(size=2),
            opacity=0.5
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data, y=mgce1_temp_rolling,
            error_y=dict(type='data', array=mgce1_temp_uncertainty_rolling, visible=True),
            mode='lines', name='MGCE1 Temp (Rolling Avg)',
            line=dict(color='green', width=3)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne DGFI1 Temp (Average)'],
            mode='lines+markers', name='DGFI1 Temperature',
            line=dict(color='blue', width=1), marker=dict(size=2),
            opacity=0.5
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data, y=dgfi1_temp_rolling,
            error_y=dict(type='data', array=dgfi1_temp_uncertainty_rolling, visible=True),
            mode='lines', name='DGFI1 Temp (Rolling Avg)',
            line=dict(color='blue', width=3)
        ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f'Time Series Overview - All Measurements (Rolling Average: {rolling_window} points)',
        height=height,
        showlegend=True,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Density [kg/m³]", row=1, col=1)
    fig.update_yaxes(title_text="Pressure [Pa]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [°C]", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    # Save file
    html_file = f"{output_prefix}_overview.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    #save_as_png(fig, f"{output_prefix}_overview")

    return fig


def create_density_plot(df, output_prefix="time_series", rolling_window=50):
    """Create density comparison plot"""
    fig = go.Figure()
    time_data = df['local_time']

    # Calculate rolling averages and uncertainties
    dma_rolling = df['DMA_Density_kg_m3'].rolling(window=rolling_window, center=True).mean()

    mgce1_rolling = df['TrueDyne MGCE1 Density (Average)'].rolling(window=rolling_window, center=True).mean()
    mgce1_uncertainty_rolling = (df['TrueDyne MGCE1 Density Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                              center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dgfi1_rolling = df['TrueDyne DGFI1 Density (Average)'].rolling(window=rolling_window, center=True).mean()
    dgfi1_uncertainty_rolling = (df['TrueDyne DGFI1 Density Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                              center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    # Original data (no error bars)
    fig.add_trace(go.Scatter(
        x=time_data, y=df['DMA_Density_kg_m3'],
        mode='lines+markers', name='DMA Density',
        line=dict(color='red', width=1), marker=dict(size=3),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Density (Average)'],
        mode='lines+markers', name='MGCE1 Density',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Density (Average)'],
        mode='lines+markers', name='DGFI1 Density',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    # Rolling averages with error bars
    fig.add_trace(go.Scatter(
        x=time_data, y=dma_rolling,
        mode='lines', name='DMA Density (Rolling Avg)',
        line=dict(color='red', width=4)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_rolling,
        error_y=dict(type='data', array=mgce1_uncertainty_rolling, visible=True),
        mode='lines', name='MGCE1 Density (Rolling Avg)',
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_rolling,
        error_y=dict(type='data', array=dgfi1_uncertainty_rolling, visible=True),
        mode='lines', name='DGFI1 Density (Rolling Avg)',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title=f'Density Comparison (Rolling Average: {rolling_window} points)',
        xaxis_title='Time',
        yaxis_title='Density [kg/m³]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_density.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    #(fig, f"{output_prefix}_overview")

    return fig


def create_pressure_plot(df, output_prefix="time_series", rolling_window=50):
    """Create pressure comparison plot"""
    fig = go.Figure()
    time_data = df['local_time']

    # Calculate rolling averages and uncertainties
    mgce1_rolling = df['TrueDyne MGCE1 Press (Average)'].rolling(window=rolling_window, center=True).mean()
    mgce1_uncertainty_rolling = (df['TrueDyne MGCE1 Press Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                            center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dgfi1_rolling = df['TrueDyne DGFI1 Press (Average)'].rolling(window=rolling_window, center=True).mean()
    dgfi1_uncertainty_rolling = (df['TrueDyne DGFI1 Press Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                            center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    # Original data (no error bars)
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press (Average)'],
        mode='lines+markers', name='MGCE1 Pressure',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press (Average)'],
        mode='lines+markers', name='DGFI1 Pressure',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    # Rolling averages with error bars
    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_rolling,
        error_y=dict(type='data', array=mgce1_uncertainty_rolling, visible=True),
        mode='lines', name='MGCE1 Pressure (Rolling Avg)',
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_rolling,
        error_y=dict(type='data', array=dgfi1_uncertainty_rolling, visible=True),
        mode='lines', name='DGFI1 Pressure (Rolling Avg)',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title=f'Pressure Comparison (Rolling Average: {rolling_window} points)',
        xaxis_title='Time',
        yaxis_title='Pressure [Pa]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_pressure.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")


    dgfi1_rolling = df['TrueDyne DGFI1 Press (Average)'].rolling(window=rolling_window, center=True).mean()

    # Original data (lighter)
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Pressure',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Pressure',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    # Rolling averages (bold)
    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_rolling,
        mode='lines', name='MGCE1 Pressure (Rolling Avg)',
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_rolling,
        mode='lines', name='DGFI1 Pressure (Rolling Avg)',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title=f'Pressure Comparison (Rolling Average: {rolling_window} points)',
        xaxis_title='Time',
        yaxis_title='Pressure [Pa]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_pressure.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    return fig

def save_as_png(fig, filename_base):
    """Save plotly figure as PNG"""
    try:
        png_file = f"{filename_base}.png"
        fig.write_image(png_file, width=1200, height=800, scale=2)
        print(f"Saved PNG: {png_file}")
    except Exception as e:
        print(f"Could not save PNG {filename_base}.png: {e}")
        print("Install kaleido with: pip install kaleido")


def create_temperature_plot(df, output_prefix="time_series", rolling_window=50):
    """Create temperature comparison plot"""
    if 'TrueDyne MGCE1 Temp (Average)' not in df.columns:
        print("No temperature data found, skipping temperature plot")
        return None

    fig = go.Figure()
    time_data = df['local_time']

    # Calculate rolling averages and uncertainties
    dma_rolling = df['T(cell)'].rolling(window=rolling_window, center=True).mean()

    mgce1_rolling = df['TrueDyne MGCE1 Temp (Average)'].rolling(window=rolling_window, center=True).mean()
    mgce1_uncertainty_rolling = (df['TrueDyne MGCE1 Temp Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                           center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    dgfi1_rolling = df['TrueDyne DGFI1 Temp (Average)'].rolling(window=rolling_window, center=True).mean()
    dgfi1_uncertainty_rolling = (df['TrueDyne DGFI1 Temp Uncertainty (std)'] ** 2).rolling(window=rolling_window,
                                                                                           center=True).mean().apply(
        lambda x: x ** 0.5 / (rolling_window ** 0.5))

    # Original data (no error bars)
    fig.add_trace(go.Scatter(
        x=time_data, y=df['T(cell)'],
        mode='lines+markers', name='DMA Cell Temperature',
        line=dict(color='red', width=1), marker=dict(size=3),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Temp (Average)'],
        mode='lines+markers', name='MGCE1 Temperature',
        line=dict(color='green', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Temp (Average)'],
        mode='lines+markers', name='DGFI1 Temperature',
        line=dict(color='blue', width=1), marker=dict(size=2),
        opacity=0.5
    ))

    # Rolling averages with error bars
    fig.add_trace(go.Scatter(
        x=time_data, y=dma_rolling,
        mode='lines', name='DMA Cell Temp (Rolling Avg)',
        line=dict(color='red', width=4)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=mgce1_rolling,
        error_y=dict(type='data', array=mgce1_uncertainty_rolling, visible=True),
        mode='lines', name='MGCE1 Temp (Rolling Avg)',
        line=dict(color='green', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=dgfi1_rolling,
        error_y=dict(type='data', array=dgfi1_uncertainty_rolling, visible=True),
        mode='lines', name='DGFI1 Temp (Rolling Avg)',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title=f'Temperature Comparison (Rolling Average: {rolling_window} points)',
        xaxis_title='Time',
        yaxis_title='Temperature [°C]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_temperature.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    #save_as_png(fig, f"{output_prefix}_overview")

    return fig


def create_uncertainty_plot(df, output_prefix="time_series"):
    """Create uncertainty analysis plot"""
    has_temp = 'TrueDyne MGCE1 Temp (Average)' in df.columns
    rows = 3 if has_temp else 2
    titles = ['Density Uncertainties', 'Pressure Uncertainties']
    if has_temp:
        titles.append('Temperature Uncertainties')

    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.1)
    time_data = df['local_time']

    # Density uncertainties
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Density Uncertainty (std)'],
        mode='lines+markers', name='MGCE1 Density Uncertainty',
        line=dict(color='green', width=2), marker=dict(size=4)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Density Uncertainty (std)'],
        mode='lines+markers', name='DGFI1 Density Uncertainty',
        line=dict(color='blue', width=2), marker=dict(size=4)
    ), row=1, col=1)

    # Pressure uncertainties
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press Uncertainty (std)'],
        mode='lines+markers', name='MGCE1 Pressure Uncertainty',
        line=dict(color='green', width=2), marker=dict(size=4)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press Uncertainty (std)'],
        mode='lines+markers', name='DGFI1 Pressure Uncertainty',
        line=dict(color='blue', width=2), marker=dict(size=4)
    ), row=2, col=1)

    # Temperature uncertainties
    if has_temp:
        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne MGCE1 Temp Uncertainty (std)'],
            mode='lines+markers', name='MGCE1 Temperature Uncertainty',
            line=dict(color='green', width=2), marker=dict(size=4)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne DGFI1 Temp Uncertainty (std)'],
            mode='lines+markers', name='DGFI1 Temperature Uncertainty',
            line=dict(color='blue', width=2), marker=dict(size=4)
        ), row=3, col=1)

    fig.update_layout(
        title='Measurement Uncertainties',
        height=800 if has_temp else 600,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Density Std [kg/m³]", row=1, col=1)
    fig.update_yaxes(title_text="Pressure Std [Pa]", row=2, col=1)
    if has_temp:
        fig.update_yaxes(title_text="Temperature Std [°C]", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
    else:
        fig.update_xaxes(title_text="Time", row=2, col=1)

    html_file = f"{output_prefix}_uncertainties.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    #save_as_png(fig, f"{output_prefix}_overview")

    return fig


def print_statistics(df):
    """Print data statistics"""
    print("\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)

    cols = df.columns.tolist()
    print(f"Available columns: {len(cols)}")
    print(f"Column names: {cols}")

    # Basic measurements
    measurements = {
        'DMA Density (g/cm³)': df['Density'],
        'DMA Density (kg/m³)': df['DMA_Density_kg_m3'],
        'DMA Temperature': df['T(cell)'],
        'MGCE1 Density': df['TrueDyne MGCE1 Density (Average)'],
        'MGCE1 Pressure': df['TrueDyne MGCE1 Press (Average)'],
        'DGFI1 Density': df['TrueDyne DGFI1 Density (Average)'],
        'DGFI1 Pressure': df['TrueDyne DGFI1 Press (Average)']
    }

    # Add temperature if available
    if 'TrueDyne MGCE1 Temp (Average)' in df.columns:
        measurements['MGCE1 Temperature'] = df['TrueDyne MGCE1 Temp (Average)']
        measurements['DGFI1 Temperature'] = df['TrueDyne DGFI1 Temp (Average)']

        # Print temperature comparison for verification
        print(f"\nTemperature data verification:")
        print(f"DMA Cell Temp range: {df['T(cell)'].min():.2f} to {df['T(cell)'].max():.2f} °C")
        print(
            f"MGCE1 Temp range: {df['TrueDyne MGCE1 Temp (Average)'].min():.2f} to {df['TrueDyne MGCE1 Temp (Average)'].max():.2f} °C")
        print(
            f"DGFI1 Temp range: {df['TrueDyne DGFI1 Temp (Average)'].min():.2f} to {df['TrueDyne DGFI1 Temp (Average)'].max():.2f} °C")

    print("\nMeasurements:")
    for name, data in measurements.items():
        print(f"{name:20}: Mean={data.mean():.4f}, Range=[{data.min():.4f}, {data.max():.4f}]")


def main():
    """Main function"""
    data_file = "fused_data_2.csv"
    output_prefix = "Rolling"
    rolling_window = 50  # Default rolling window size

    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_prefix = sys.argv[2]
    if len(sys.argv) >= 4:
        rolling_window = int(sys.argv[3])

    print("=== Time Series Plotter ===")
    print(f"Data file: {data_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Rolling window: {rolling_window} points")

    try:
        df = load_and_parse_data(data_file)
        print_statistics(df)

        print("\nCreating plots...")
        create_overview_plot(df, output_prefix, rolling_window)
        create_density_plot(df, output_prefix, rolling_window)
        create_pressure_plot(df, output_prefix, rolling_window)
        create_temperature_plot(df, output_prefix, rolling_window)
        create_uncertainty_plot(df, output_prefix)

        print("\nDone! Open the HTML files in your browser.")

    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()