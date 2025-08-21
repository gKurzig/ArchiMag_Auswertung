import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_measurements(json_file='AllScaleMeasurements.json'):
    """
    Load measurements from JSON file.

    Args:
        json_file (str): Path to the JSON file

    Returns:
        dict: Dictionary containing all measurements
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file}': {str(e)}")
        return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None


def calculate_averages_with_std(scale_weights, window_size=500):
    """
    Calculate moving averages and standard deviations for scale weights.

    Args:
        scale_weights (list): List of scale weight values
        window_size (int): Number of measurements to average (default: 500)

    Returns:
        tuple: (averages, std_devs, measurement_indices)
    """
    # Convert to numpy array and ensure numeric values
    try:
        weights_array = np.array(scale_weights, dtype=float)
    except (ValueError, TypeError) as e:
        print(f"Error converting scale weights to numeric: {e}")
        print(f"Sample problematic data: {scale_weights[:10]}")
        return [], [], []

    # Remove any NaN values
    weights_array = weights_array[~np.isnan(weights_array)]

    if len(weights_array) == 0:
        print("Warning: No valid numeric data found")
        return [], [], []

    if len(weights_array) < window_size:
        print(f"Warning: Only {len(weights_array)} measurements available, less than window size {window_size}")
        # If we have fewer measurements than window size, use all available data
        avg = np.mean(weights_array)
        std = np.std(weights_array)
        return [avg], [std], [len(weights_array) // 2]

    averages = []
    std_devs = []
    measurement_indices = []

    # Calculate averages for every 500 measurements
    for i in range(0, len(weights_array), window_size):
        end_idx = min(i + window_size, len(weights_array))
        window_data = weights_array[i:end_idx]

        if len(window_data) > 0:  # Make sure we have data
            avg = np.mean(window_data)
            std = np.std(window_data)

            averages.append(avg)
            std_devs.append(std)
            measurement_indices.append(i + len(window_data) // 2)  # Middle point of the window

    return averages, std_devs, measurement_indices


def create_plot(measurements_data):
    """
    Create a plotly plot with all measurements and interactive controls.

    Args:
        measurements_data (dict): Dictionary containing all measurement data
    """
    fig = go.Figure()

    # Color palette for different measurements
    colors = px.colors.qualitative.Set1
    color_idx = 0

    # Store trace information for buttons
    trace_info = []

    for file_id, data in measurements_data.items():
        # Get scale weights
        scale_weights = data.get('Scale Weight [g]', [])

        if not scale_weights:
            print(f"Warning: No scale weight data found for {file_id}")
            continue

        # Calculate averages and standard deviations
        averages, std_devs, indices = calculate_averages_with_std(scale_weights)

        # Normalize by subtracting the overall average of this measurement
        # Convert to numpy array to ensure proper numeric handling
        averages_array = np.array(averages, dtype=float)
        overall_average = np.mean(averages_array)

        # Check for NaN
        if np.isnan(overall_average):
            print(f"Warning: NaN detected in averages for {file_id}")
            print(f"First few averages: {averages[:5]}")
            print(f"Scale weights sample: {scale_weights[:10]}")
            continue

        normalized_averages = averages_array - overall_average

        # Create legend label
        description = data.get('Description', 'No description')
        legend_label = f"{file_id}: {description}"
        short_label = f"{file_id}"

        # Limit legend text length for readability
        if len(legend_label) > 60:
            legend_label = legend_label[:57] + "..."

        # Get color for this trace
        current_color = colors[color_idx % len(colors)]

        # Add the main trace (normalized averages)
        fig.add_trace(go.Scatter(
            x=indices,
            y=normalized_averages,
            mode='lines+markers',
            name=legend_label,
            line=dict(color=current_color, width=2),
            marker=dict(size=6, color=current_color),
            showlegend=True,
            legendgroup=short_label,  # Group with error bars
        ))

        # Add error bars (standard deviation) - also normalized
        fig.add_trace(go.Scatter(
            x=indices + indices[::-1],  # x, then x reversed
            y=[avg + std for avg, std in zip(normalized_averages, std_devs)] +
              [avg - std for avg, std in zip(normalized_averages[::-1], std_devs[::-1])],  # upper, then lower reversed
            fill='toself',
            fillcolor=f'rgba({current_color[4:-1]}, 0.2)',  # Convert to rgba with transparency
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name=f'{legend_label} ±1σ',
            legendgroup=short_label,  # Group with main trace
        ))

        # Store trace info for buttons
        trace_info.append({
            'file_id': file_id,
            'description': description,
            'short_label': short_label,
            'legend_label': legend_label,
            'main_trace_index': len(fig.data) - 2,  # Index of main trace
            'error_trace_index': len(fig.data) - 1  # Index of error trace
        })

        color_idx += 1

        print(
            f"Processed {file_id}: {len(scale_weights)} measurements -> {len(averages)} averages (normalized, mean offset: {overall_average:.3f}g)")

    # Create dropdown buttons for showing/hiding traces
    buttons = []

    # "Show All" button
    show_all_visibility = [True] * len(fig.data)
    buttons.append(dict(
        label="Show All",
        method="update",
        args=[{"visible": show_all_visibility}]
    ))

    # "Hide All" button
    hide_all_visibility = [False] * len(fig.data)
    buttons.append(dict(
        label="Hide All",
        method="update",
        args=[{"visible": hide_all_visibility}]
    ))

    # Individual toggle buttons for each measurement
    for trace in trace_info:
        # Create visibility array - all current visibility except toggle this trace
        current_visibility = [True] * len(fig.data)
        current_visibility[trace['main_trace_index']] = 'legendonly'
        current_visibility[trace['error_trace_index']] = 'legendonly'

        buttons.append(dict(
            label=f"Toggle {trace['short_label']}",
            method="update",
            args=[{"visible": current_visibility}]
        ))

    # Update layout with interactive controls
    fig.update_layout(
        title={
            'text': 'Normalized Scale Weight Measurements - Interactive Plot<br><sub>Each measurement normalized by subtracting its mean value</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Measurement Index',
        yaxis_title='Normalized Scale Weight [g] (relative to mean)',
        hovermode='x unified',

        # Interactive dropdown menu
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            )
        ],

        # Legend configuration
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            itemclick="toggle",  # Allow clicking to toggle
            itemdoubleclick="toggleothers"  # Double-click to show only this one
        ),

        margin=dict(r=350, t=80),  # Make room for legend and title
        width=1300,
        height=650,
        template='plotly_white',

        # Add annotations for instructions
        annotations=[
            dict(
                text="💡 Click legend items to show/hide<br>Double-click to isolate<br>Use dropdown for bulk operations",
                showarrow=False,
                xref="paper", yref="paper",
                x=1.02, y=0.02,
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1
            )
        ]
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)'
    )

    return fig


def main():
    """
    Main function to load data and create plot.
    """
    # Load the JSON data
    json_file = input("Enter the JSON file path (or press Enter for 'AllScaleMeasurements.json'): ").strip()
    if not json_file:
        json_file = 'AllScaleMeasurements.json'

    measurements_data = load_measurements(json_file)
    if not measurements_data:
        return

    print(f"Found {len(measurements_data)} measurement files")

    # Create the plot
    fig = create_plot(measurements_data)

    # Automatically save as HTML
    html_filename = 'AllScales.html'
    fig.write_html(html_filename)
    print(f"Plot automatically saved as {html_filename}")
    print("Each measurement has been normalized by subtracting its individual mean value.")


if __name__ == "__main__":
    main()