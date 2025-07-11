import os
from pathlib import Path

from fusion import fuse_data_no_na
from file_selector import main as file_selector_main

# Add these instead:
DATA_FOLDER = None
DMA_FILE = None
TRUEDYNE_FILE = None
OUTPUT_FILE = "fused_data.csv"


def getFiles():
    global DATA_FOLDER, DMA_FILE, TRUEDYNE_FILE, OUTPUT_FILE



    # Use file_selector to get configuration
    config = file_selector_main()

    if config:
        DATA_FOLDER = config['DATA_FOLDER']
        DMA_FILE = config['DMA_PATH']
        TRUEDYNE_FILE = config['TRUEDYNE_PATH']
        OUTPUT_FILE = config['OUTPUT_PATH']

        print(f"Configuration loaded:")
        print(f"DATA_FOLDER: {DATA_FOLDER}")
        print(f"DMA_FILE: {DMA_FILE}")
        print(f"TRUEDYNE_FILE: {TRUEDYNE_FILE}")
        print(f"OUTPUT_FILE: {OUTPUT_FILE}")
    else:
        print("Failed to load configuration")
        return False

    return True


def fuse_dma_TrueDyne():
    print(f"Running fusion...")
    print(f"DMA file: {DMA_FILE}")
    print(f"TrueDyne file: {TRUEDYNE_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # Run fusion with full paths
    fused_data = fuse_data_no_na(DMA_FILE, TRUEDYNE_FILE, OUTPUT_FILE)

    if fused_data is not None:
        print("\nFirst 5 rows:")
        print(fused_data.head().to_string())
        return True
    return False


def plotting():
    """Run plotting functions on existing fused data"""
    # Change to data directory
    original_dir = os.getcwd()
    os.chdir(DATA_FOLDER)

    try:
        print("=== Running Plotting Functions ===")

        # Create plots folder in the same location as measurement data
        plots_folder = os.path.join(DATA_FOLDER, "plots")
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            print(f"Created plots folder: {plots_folder}")

        # Change to plots folder
        os.chdir(plots_folder)

        # # Create plots folder
        # plots_folder = "plots"
        # if not os.path.exists(plots_folder):
        #     os.makedirs(plots_folder)
        #     print(f"Created plots folder: {plots_folder}")

        # Change to plots folder
        os.chdir(plots_folder)

        from Plots_rolling import load_and_parse_data, create_overview_plot, create_density_plot, create_pressure_plot, \
            create_temperature_plot, create_uncertainty_plot, print_statistics,create_density_centered_plot

        # Load the output file from parent directory
        #plot_df = load_and_parse_data(f"../{OUTPUT_FILE}")
        plot_df = load_and_parse_data(OUTPUT_FILE)
        print_statistics(plot_df)

        print("\nCreating plots...")
        rolling_window = 50  # Adjust as needed
        create_overview_plot(plot_df, "fused_plots", rolling_window)
        create_density_plot(plot_df, "fused_plots", rolling_window)
        create_pressure_plot(plot_df, "fused_plots", rolling_window)
        create_temperature_plot(plot_df, "fused_plots", rolling_window)
        create_uncertainty_plot(plot_df, "fused_plots")
        create_density_centered_plot(plot_df, "fused_plots", rolling_window)


        # print(f"Plots saved in: {DATA_FOLDER}/plots/")
        print(f"Plots saved in: {plots_folder}")

        print("Open the HTML files in your browser.")

    finally:
        os.chdir(original_dir)


def correlation_analysis():
    """Run correlation analysis on existing fused data"""
    # Change to data directory
    original_dir = os.getcwd()
    os.chdir(DATA_FOLDER)

    try:
        print("=== Running Correlation Analysis ===")



        # Create correlation folder in the same location as measurement data
        correlation_folder = os.path.join(DATA_FOLDER, "correlation")
        if not os.path.exists(correlation_folder):
            os.makedirs(correlation_folder)
            print(f"Created correlation folder: {correlation_folder}")

        # Change to correlation folder
        os.chdir(correlation_folder)


        from Correlation import (load_and_prepare_data, create_density_timeseries_plot_seperat,
                                 create_dma_truedyne_rolling_difference_plot,
                                 create_dma_total_avg_truedyne_rolling_difference_plot,
                                 plot_three_densities_with_averages,print_unexplainable_difference_report,
                                 plot_offset_corrected_comparison)

        # Load the output file from parent directory
        #df, df_analysis, numerical_cols = load_and_prepare_data(f"../{OUTPUT_FILE}")
        df, df_analysis, numerical_cols = load_and_prepare_data(OUTPUT_FILE)
        # Add this to the correlation_analysis() function
        print_unexplainable_difference_report(df)
        #plot_offset_corrected_comparison(df, 'MGCE1','DGFI1')
        #plot_offset_corrected_comparison(df, 'DGFI1')
        plot_offset_corrected_comparison(df, 'MGCE1','DGFI1', rolling_window=10)


        # print("\nCreating correlation plots...")
        # print(df.columns)
        # print(df_analysis.columns)
        # window_size = 50
        #
        # # # Create the main plots
        # create_density_timeseries_plot_seperat(df, df_analysis, window_size)
        # create_dma_truedyne_rolling_difference_plot(df, df_analysis, instrument='MGCE1', window_size=window_size)
        # create_dma_truedyne_rolling_difference_plot(df, df_analysis, instrument='DGFI1', window_size=window_size)
        #
        # create_dma_total_avg_truedyne_rolling_difference_plot(df, df_analysis, instrument='MGCE1',window_size=window_size)
        # create_dma_total_avg_truedyne_rolling_difference_plot(df, df_analysis, instrument='DGFI1', window_size=window_size)
        #
        # plot_three_densities_with_averages(df_analysis)
        # #
        # print(f"Correlation analysis saved in: {DATA_FOLDER}/correlation/")
        print(f"Correlation analysis saved in: {correlation_folder}")
        print("Open the HTML files in your browser.")

    finally:
        os.chdir(original_dir)



if __name__ == "__main__":
    if getFiles():
        success = fuse_dma_TrueDyne()
        if success and os.path.exists(OUTPUT_FILE):
            plotting()
            correlation_analysis()
        else:
            print("Fusion failed or no output file created. Skipping plots.")
    else:
        print("Exiting due to configuration error")