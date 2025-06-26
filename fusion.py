import pandas as pd
import numpy as np
from datetime import datetime
import sys


def parse_timestamps(df, timestamp_col):
    """
    Parse timestamps from different formats to datetime objects
    """
    try:
        if timestamp_col == 'Date of Measurement':
            # DMA format - assume local CET/CEST time, convert to UTC
            timestamps = pd.to_datetime(df[timestamp_col])
            timestamps = timestamps.dt.tz_localize('Europe/Vienna').dt.tz_convert('UTC')
        else:
            # TrueDyne format - assume local CET/CEST time, convert to UTC
            timestamps = pd.to_datetime(df[timestamp_col])
            timestamps = timestamps.dt.tz_localize('Europe/Vienna').dt.tz_convert('UTC')
        return timestamps
    except Exception as e:
        print(f"Error parsing timestamps in column {timestamp_col}: {e}")
        # Fallback to manual parsing
        timestamps = []
        for ts in df[timestamp_col]:
            if pd.isna(ts):
                timestamps.append(pd.NaT)
            elif isinstance(ts, str):
                try:
                    if 'T' in ts and ('+' in ts or 'Z' in ts):
                        dt = pd.to_datetime(ts, utc=True)
                    else:
                        dt = pd.to_datetime(ts).tz_localize('UTC')
                    timestamps.append(dt)
                except:
                    timestamps.append(pd.NaT)
            else:
                timestamps.append(pd.NaT)
        return pd.Series(timestamps)


def calculate_stats(values):
    """
    Calculate average and standard deviation for a list of values
    Handle missing values by dropping them
    """
    if len(values) == 0:
        return np.nan, np.nan

    # Remove NaN, None, and invalid values
    clean_values = []
    for v in values:
        if pd.notna(v) and v is not None:
            try:
                float_val = float(v)
                if not np.isinf(float_val):
                    clean_values.append(float_val)
            except (ValueError, TypeError):
                continue

    if len(clean_values) == 0:
        return np.nan, np.nan

    avg = np.mean(clean_values)
    std = np.std(clean_values, ddof=1) if len(clean_values) > 1 else 0.0
    return avg, std


def fuse_data_no_na(dma_file, truedyne_file, output_file):
    """
    Fuse DMA and TrueDyne data based on timestamps
    Only include DMA records that have corresponding TrueDyne data (no NaN values)
    """
    try:
        # Read the CSV files
        print(f"Reading DMA data from {dma_file}...")
        #dma_df = pd.read_csv(dma_file, delimiter=';')
        dma_df = pd.read_csv(dma_file, delimiter=';', encoding='latin-1')

        print(f"Reading TrueDyne data from {truedyne_file}...")
        #truedyne_df = pd.read_csv(truedyne_file, delimiter=',')
        truedyne_df = pd.read_csv(truedyne_file, delimiter=',', encoding='latin-1')

        print(f"DMA data shape: {dma_df.shape}")
        print(f"TrueDyne data shape: {truedyne_df.shape}")

        # Parse timestamps
        print("Parsing timestamps...")
        dma_df['parsed_timestamp'] = parse_timestamps(dma_df, 'Date of Measurement')
        truedyne_df['parsed_timestamp'] = parse_timestamps(truedyne_df, 'Timestamp')

        # Remove rows with invalid timestamps
        dma_df = dma_df.dropna(subset=['parsed_timestamp'])
        truedyne_df = truedyne_df.dropna(subset=['parsed_timestamp'])

        # Drop TrueDyne rows with missing critical values
        print("Cleaning TrueDyne data - removing rows with missing values...")
        initial_truedyne_count = len(truedyne_df)

        # Define critical columns that should not have missing values
        critical_columns = [
            'TrueDyne MGCE1 Density [kg/m**3]',
            'TrueDyne MGCE1 Press [Pa]',
            'TrueDyne MGCE1 Temp [C]',
            'TrueDyne DGFI1 Density [kg/m**3]',
            'TrueDyne DGFI1 Press [Pa]',
            'TrueDyne DGFI1 Temp [C]'
        ]

        # Drop rows where any critical column has missing values
        truedyne_df = truedyne_df.dropna(subset=critical_columns)

        cleaned_count = len(truedyne_df)
        dropped_count = initial_truedyne_count - cleaned_count

        if dropped_count > 0:
            print(f"Dropped {dropped_count} TrueDyne measurements due to missing values")

        # Sort by timestamp
        dma_df = dma_df.sort_values('parsed_timestamp')
        truedyne_df = truedyne_df.sort_values('parsed_timestamp')

        print(f"Valid DMA records: {len(dma_df)}")
        print(f"Valid TrueDyne records: {len(truedyne_df)}")

        # Check time overlap
        dma_start = dma_df['parsed_timestamp'].min()
        dma_end = dma_df['parsed_timestamp'].max()
        truedyne_start = truedyne_df['parsed_timestamp'].min()
        truedyne_end = truedyne_df['parsed_timestamp'].max()

        print(f"\nTime ranges:")
        print(f"DMA: {dma_start} to {dma_end}")
        print(f"TrueDyne: {truedyne_start} to {truedyne_end}")

        # Check for overlap
        overlap_start = max(dma_start, truedyne_start)
        overlap_end = min(dma_end, truedyne_end)

        if overlap_start >= overlap_end:
            print(f"\nWARNING: No time overlap between datasets!")
            print(f"All DMA records will be skipped due to no corresponding TrueDyne data.")
        else:
            print(f"\nTime overlap: {overlap_start} to {overlap_end}")

        # Create result list
        results = []
        skipped_count = 0

        # Process each DMA measurement
        for i, dma_row in dma_df.iterrows():
            dma_time = dma_row['parsed_timestamp']

            # Find the time window for averaging TrueDyne data
            if i == 0:
                # For the first DMA measurement, use all TrueDyne data before this timestamp
                start_time = truedyne_df['parsed_timestamp'].min()
            else:
                # Use the previous DMA timestamp as start time
                prev_idx = dma_df.index.get_loc(i) - 1
                prev_dma_time = dma_df.iloc[prev_idx]['parsed_timestamp']
                start_time = prev_dma_time

            end_time = dma_time

            # Filter TrueDyne data for this time window
            mask = (truedyne_df['parsed_timestamp'] > start_time) & (truedyne_df['parsed_timestamp'] <= end_time)
            window_data = truedyne_df[mask]

            # Skip DMA records with no corresponding TrueDyne data
            if len(window_data) == 0:
                print(f"Skipping DMA record {dma_row['Sample Number']} at {dma_time} - no TrueDyne data")
                skipped_count += 1
                continue

            #print(f"Processing DMA record {dma_row['Sample Number']} at {dma_time}")
            #print(f"  Time window: {start_time} to {end_time}")
            #print(f"  TrueDyne measurements in window: {len(window_data)}")

            # Calculate statistics for MGCE1
            mgce1_density_avg, mgce1_density_std = calculate_stats(
                window_data['TrueDyne MGCE1 Density [kg/m**3]'].tolist())
            mgce1_press_avg, mgce1_press_std = calculate_stats(window_data['TrueDyne MGCE1 Press [Pa]'].tolist())
            mgce1_temp_avg, mgce1_temp_std = calculate_stats(window_data['TrueDyne MGCE1 Temp [C]'].tolist())

            # Calculate statistics for DGFI1
            dgfi1_density_avg, dgfi1_density_std = calculate_stats(
                window_data['TrueDyne DGFI1 Density [kg/m**3]'].tolist())
            dgfi1_press_avg, dgfi1_press_std = calculate_stats(window_data['TrueDyne DGFI1 Press [Pa]'].tolist())
            dgfi1_temp_avg, dgfi1_temp_std = calculate_stats(window_data['TrueDyne DGFI1 Temp [C]'].tolist())

            # Only include if we have valid measurements (no NaN)
            if (pd.notna(mgce1_density_avg) and pd.notna(mgce1_press_avg) and pd.notna(mgce1_temp_avg) and
                    pd.notna(dgfi1_density_avg) and pd.notna(dgfi1_press_avg) and pd.notna(dgfi1_temp_avg)):

                # Create result row
                result_row = {
                    'Number': dma_row['Sample Number'],
                    'Timestamp': dma_row['Date of Measurement'],
                    'T(cell)': dma_row['T (cell)'],
                    'Density': dma_row['Density'],
                    'TrueDyne MGCE1 Density (Average)': mgce1_density_avg,
                    'TrueDyne MGCE1 Density Uncertainty (std)': mgce1_density_std,
                    'TrueDyne MGCE1 Press (Average)': mgce1_press_avg,
                    'TrueDyne MGCE1 Press Uncertainty (std)': mgce1_press_std,
                    'TrueDyne MGCE1 Temp (Average)': mgce1_temp_avg,
                    'TrueDyne MGCE1 Temp Uncertainty (std)': mgce1_temp_std,
                    'TrueDyne DGFI1 Density (Average)': dgfi1_density_avg,
                    'TrueDyne DGFI1 Density Uncertainty (std)': dgfi1_density_std,
                    'TrueDyne DGFI1 Press (Average)': dgfi1_press_avg,
                    'TrueDyne DGFI1 Press Uncertainty (std)': dgfi1_press_std,
                    'TrueDyne DGFI1 Temp (Average)': dgfi1_temp_avg,
                    'TrueDyne DGFI1 Temp Uncertainty (std)': dgfi1_temp_std
                }

                results.append(result_row)
            else:
                print(f"  Skipping due to invalid TrueDyne calculations")
                skipped_count += 1

        # Create output DataFrame
        result_df = pd.DataFrame(results)

        if len(result_df) == 0:
            print("\nWARNING: No valid fused records created!")
            print("This likely means there's no time overlap between your datasets.")
            print("Check your timestamp formats and time ranges.")
            return None

        # Save to CSV
        print(f"\nSaving fused data to {output_file}...")
        result_df.to_csv(output_file, index=False)

        print(f"Successfully created fused dataset with {len(result_df)} records")
        print(f"Skipped {skipped_count} DMA records due to missing TrueDyne data")
        print(f"Output saved to: {output_file}")

        # Display summary statistics
        print("\nSummary:")
        print(f"Original DMA records: {len(dma_df)}")
        print(f"Original TrueDyne records: {len(truedyne_df)}")
        print(f"Fused records (no NaN): {len(result_df)}")
        print(f"Skipped records: {skipped_count}")

        return result_df

    except Exception as e:
        print(f"Error during data fusion: {str(e)}")
        raise


def main():
    """
    Main function to run the data fusion process
    """
    # Default file names (can be modified or passed as command line arguments)
    dma_file = "dma_messdaten_MessReihe_2.csv"
    truedyne_file = "DMA_Results_20250612_181930_truedyne_buffer.csv"
    output_file = "fused_data_3.csv"

    # Allow command line arguments
    if len(sys.argv) >= 4:
        dma_file = sys.argv[1]
        truedyne_file = sys.argv[2]
        output_file = sys.argv[3]
    elif len(sys.argv) >= 2:
        print("Usage: python script.py [dma_file] [truedyne_file] [output_file]")
        print("Using default filenames...")

    print("=== CSV Data Fusion Tool (No NaN Values) ===")
    print(f"DMA file: {dma_file}")
    print(f"TrueDyne file: {truedyne_file}")
    print(f"Output file: {output_file}")
    print("\nThis version will only include DMA records that have corresponding TrueDyne data.")
    print()

    try:
        fused_data = fuse_data_no_na(dma_file, truedyne_file, output_file)

        if fused_data is not None:
            # Display first few rows of the result
            print("\nFirst 5 rows of fused data:")
            print(fused_data.head().to_string())

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure the input files exist in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()