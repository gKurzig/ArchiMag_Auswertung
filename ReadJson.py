import json
import csv
import os
from pathlib import Path
import glob


def extract_sensor_data(json_file_path):
    """
    Extract sensor data from a single JSON file.
    Returns a dictionary with all the sensor readings.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Extract timestamp
        timestamp = data.get('timestamp', '')

        # Extract DMA4501 data
        dma_data = data.get('dma4501', {})
        dma_temperature = dma_data.get('temperature', '')
        dma_density = dma_data.get('density', '')

        # Extract TrueDyne data
        truedyne_data = data.get('truedyne', {}).get('data', {})

        # MGCE1 sensor data
        mgce1_data = truedyne_data.get('MGCE1', {})
        mgce1_density = mgce1_data.get('density', '')
        mgce1_pressure = mgce1_data.get('pressure', '')
        mgce1_temperature = mgce1_data.get('temperature', '')

        # DGFI1 sensor data
        dgfi1_data = truedyne_data.get('DGFI1', {})
        dgfi1_density = dgfi1_data.get('density', '')
        dgfi1_pressure = dgfi1_data.get('pressure', '')
        dgfi1_temperature = dgfi1_data.get('temperature', '')

        return {
            'timestamp': timestamp,
            'dma_temperature': dma_temperature,
            'dma_density': dma_density,
            'mgce1_density': mgce1_density,
            'mgce1_pressure': mgce1_pressure,
            'mgce1_temperature': mgce1_temperature,
            'dgfi1_density': dgfi1_density,
            'dgfi1_pressure': dgfi1_pressure,
            'dgfi1_temperature': dgfi1_temperature
        }

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error processing file {json_file_path}: {e}")
        return None


def process_folder_to_csv(folder_path, output_csv_path):
    """
    Process all JSON files in a folder and save to CSV.
    """
    # Define CSV header
    header = [
        'Number',
        'Timestamp',
        'DMA Temperatur',
        'DMA Density',
        'TrueDyne MGCE1 Density',
        'TrueDyne MGCE1 Press',
        'TrueDyne MGCE1 Temp',
        'TrueDyne DGFI1 Density',
        'TrueDyne DGFI1 Press',
        'TrueDyne DGFI1 Temp'
    ]

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    json_files.sort()  # Sort files for consistent ordering

    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    # Process files and write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i, json_file in enumerate(json_files, 1):
            print(f"Processing file {i}/{len(json_files)}: {os.path.basename(json_file)}")

            sensor_data = extract_sensor_data(json_file)

            if sensor_data:
                row = [
                    i,  # Number (sequential)
                    sensor_data['timestamp'],
                    sensor_data['dma_temperature'],
                    sensor_data['dma_density'],
                    sensor_data['mgce1_density'],
                    sensor_data['mgce1_pressure'],
                    sensor_data['mgce1_temperature'],
                    sensor_data['dgfi1_density'],
                    sensor_data['dgfi1_pressure'],
                    sensor_data['dgfi1_temperature']
                ]
                writer.writerow(row)
            else:
                # Write row with empty values if file couldn't be processed
                row = [i, '', '', '', '', '', '', '', '', '']
                writer.writerow(row)

    print(f"CSV file created: {output_csv_path}")


def main():
    # Configuration - modify these paths as needed
    folder_path = input("Enter the path to the folder containing JSON files: ").strip()

    # Remove quotes if user pasted a path with quotes
    folder_path = folder_path.strip('"\'')

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return

    # Generate output filename
    folder_name = os.path.basename(folder_path.rstrip('/\\'))
    output_csv_path = os.path.join(folder_path, f"{folder_name}_sensor_data.csv")

    # Ask user if they want to specify a different output path
    custom_output = input(
        f"Output CSV will be saved as: {output_csv_path}\nPress Enter to continue, or type a new path: ").strip()

    if custom_output:
        output_csv_path = custom_output.strip('"\'')

    # Process the files
    process_folder_to_csv(folder_path, output_csv_path)


if __name__ == "__main__":
    main()