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
###################
def create_dma_truedyne_difference_plot(df, df_analysis):
    """Create time series plot of DMA vs average TrueDyne difference."""
    print("Creating DMA vs TrueDyne difference plot...")
    try:
        # Calculate average of both TrueDyne density measurements
        df_analysis['TrueDyne_Avg_Density'] = (df_analysis['MGCE1_Density'] + df_analysis['DGFI1_Density']) / 2

        # Calculate difference: DMA - TrueDyne_Average
        df_analysis['DMA_TrueDyne_Diff'] = df_analysis['Density'] - df_analysis['TrueDyne_Avg_Density']

        # Create the plot
        fig = go.Figure()

        # Add the difference time series
        fig.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df_analysis['DMA_TrueDyne_Diff'],
            mode='lines+markers',
            name='DMA - TrueDyne Average',
            line=dict(color='blue', width=2),
            marker=dict(size=3)
        ))

        # Add a horizontal line at zero for reference
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Zero Difference")

        # Update layout
        fig.update_layout(
            title="DMA Density vs TrueDyne Average Difference Over Time",
            xaxis_title="Time",
            yaxis_title="Density Difference (DMA - TrueDyne Avg)",
            width=1000,
            height=600,
            showlegend=True
        )

        # Save the plot
        fig.write_html("dma_truedyne_difference.html")
        print("DMA vs TrueDyne difference plot saved as dma_truedyne_difference.html")

        # Print some statistics
        print(f"Mean difference: {df_analysis['DMA_TrueDyne_Diff'].mean():.6f}")
        print(f"Std deviation: {df_analysis['DMA_TrueDyne_Diff'].std():.6f}")
        print(f"Range: [{df_analysis['DMA_TrueDyne_Diff'].min():.6f}, {df_analysis['DMA_TrueDyne_Diff'].max():.6f}]")

    except Exception as e:
        print(f"Error creating DMA vs TrueDyne difference plot: {e}")

################################################################

def create_overview_plot(df, output_prefix="time_series"):
    """Create overview plot with all measurements"""
    has_temp = 'TrueDyne MGCE1 Temp (Average)' in df.columns

    # Always use 3 subplots: Density, Pressure, Temperature
    rows = 3
    titles = ('Density Measurements', 'Pressure Measurements', 'Temperature Measurements')
    height = 900

    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.08)
    time_data = df['local_time']

    # Density plot
    fig.add_trace(go.Scatter(
        x=time_data, y=df['DMA_Density_kg_m3'],
        mode='lines+markers', name='DMA Density',
        line=dict(color='red', width=2), marker=dict(size=6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Density (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Density Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Density',
        line=dict(color='green', width=1.5), marker=dict(size=4)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Density (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Density Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Density',
        line=dict(color='blue', width=1.5), marker=dict(size=4)
    ), row=1, col=1)

    # Pressure plot
    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Pressure',
        line=dict(color='green', width=1.5), marker=dict(size=4)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Pressure',
        line=dict(color='blue', width=1.5), marker=dict(size=4)
    ), row=2, col=1)

    # Temperature plot - ALL temperatures in one subplot
    # DMA Cell Temperature
    fig.add_trace(go.Scatter(
        x=time_data, y=df['T(cell)'],
        mode='lines+markers', name='DMA Cell Temperature',
        line=dict(color='red', width=2), marker=dict(size=4)
    ), row=3, col=1)

    # TrueDyne temperatures (if available)
    if has_temp:
        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne MGCE1 Temp (Average)'],
            error_y=dict(type='data', array=df['TrueDyne MGCE1 Temp Uncertainty (std)'], visible=True),
            mode='lines+markers', name='MGCE1 Temperature',
            line=dict(color='green', width=1.5), marker=dict(size=4)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data, y=df['TrueDyne DGFI1 Temp (Average)'],
            error_y=dict(type='data', array=df['TrueDyne DGFI1 Temp Uncertainty (std)'], visible=True),
            mode='lines+markers', name='DGFI1 Temperature',
            line=dict(color='blue', width=1.5), marker=dict(size=4)
        ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Time Series Overview - All Measurements',
        height=height,
        showlegend=True,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Density [kg/m³]", row=1, col=1)
    fig.update_yaxes(title_text="Pressure [bar]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [°C]", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    # Save file
    html_file = f"{output_prefix}_overview.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    return fig


def create_density_plot(df, output_prefix="time_series"):
    """Create density comparison plot"""
    fig = go.Figure()
    time_data = df['local_time']

    fig.add_trace(go.Scatter(
        x=time_data, y=df['DMA_Density_kg_m3'],
        mode='lines+markers', name='DMA Density',
        line=dict(color='red', width=3), marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Density (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Density Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Density',
        line=dict(color='green', width=2), marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Density (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Density Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Density',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ))

    fig.update_layout(
        title='Density Comparison',
        xaxis_title='Time',
        yaxis_title='Density [kg/m³]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_density.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    return fig


def create_pressure_plot(df, output_prefix="time_series"):
    """Create pressure comparison plot"""
    fig = go.Figure()
    time_data = df['local_time']

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Pressure',
        line=dict(color='green', width=2), marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Press (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Press Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Pressure',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ))

    fig.update_layout(
        title='Pressure Comparison',
        xaxis_title='Time',
        yaxis_title='Pressure [Pa]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_pressure.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

    return fig


def create_temperature_plot(df, output_prefix="time_series"):
    """Create temperature comparison plot"""
    if 'TrueDyne MGCE1 Temp (Average)' not in df.columns:
        print("No temperature data found, skipping temperature plot")
        return None

    fig = go.Figure()
    time_data = df['local_time']

    fig.add_trace(go.Scatter(
        x=time_data, y=df['T(cell)'],
        mode='lines+markers', name='DMA Cell Temperature',
        line=dict(color='red', width=3), marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne MGCE1 Temp (Average)'],
        error_y=dict(type='data', array=df['TrueDyne MGCE1 Temp Uncertainty (std)'], visible=True),
        mode='lines+markers', name='MGCE1 Temperature',
        line=dict(color='green', width=2), marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=time_data, y=df['TrueDyne DGFI1 Temp (Average)'],
        error_y=dict(type='data', array=df['TrueDyne DGFI1 Temp Uncertainty (std)'], visible=True),
        mode='lines+markers', name='DGFI1 Temperature',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ))

    fig.update_layout(
        title='Temperature Comparison',
        xaxis_title='Time',
        yaxis_title='Temperature [°C]',
        height=600,
        template='plotly_white'
    )

    html_file = f"{output_prefix}_temperature.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")

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

    return fig


def print_statistics(df):
    """Print data statistics"""
    print("\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)

    cols = df.columns.tolist()
    print(f"Available columns: {len(cols)}")

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

    print("\nMeasurements:")
    for name, data in measurements.items():
        print(f"{name:20}: Mean={data.mean():.4f}, Range=[{data.min():.4f}, {data.max():.4f}]")
#######################################################


def main():
    """Main function"""
    data_file = "fused_data_3.csv"
    output_prefix = "time_series_3"

    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_prefix = sys.argv[2]

    print("=== Time Series Plotter ===")
    print(f"Data file: {data_file}")
    print(f"Output prefix: {output_prefix}")

    try:
        df = load_and_parse_data(data_file)
        print_statistics(df)

        print("\nCreating plots...")
        create_overview_plot(df, output_prefix)
        create_density_plot(df, output_prefix)
        create_pressure_plot(df, output_prefix)
        create_temperature_plot(df, output_prefix)
        create_uncertainty_plot(df, output_prefix)

        print("\nDone! Open the HTML files in your browser.")

    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()