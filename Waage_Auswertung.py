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


def main():
    """
    Main function to run the script.
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


if __name__ == "__main__":
    main()