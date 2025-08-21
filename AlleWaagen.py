import os
import pandas as pd
import json
import re
from pathlib import Path


def extract_measurement_description(folder_path):
    """
    Extract measurement description from measurement_description.txt file.

    Args:
        folder_path (str): Path to the folder containing the CSV file

    Returns:
        str or None: Description text or None if file doesn't exist
    """
    # Look for measurement_description.txt in subfolders
    for root, dirs, files in os.walk(folder_path):
        if 'measurement_description.txt' in files:
            desc_file_path = os.path.join(root, 'measurement_description.txt')
            try:
                with open(desc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract the description part between the === lines
                lines = content.split('\n')
                description_lines = []
                in_description = False

                for line in lines:
                    if line.strip() == '========================================':
                        if not in_description:
                            in_description = True
                            continue
                        else:
                            break
                    elif in_description and line.strip() != 'End of description':
                        description_lines.append(line.strip())

                description = '\n'.join(description_lines).strip()
                if description:
                    print(f"Found description: {description[:50]}...")
                    return description

            except Exception as e:
                print(f"Error reading description file {desc_file_path}: {str(e)}")

    return None


def extract_scale_measurements(root_folder):
    """
    Extract scale measurements from CSV files matching the pattern
    DMA_Results_*_truedyne_buffer.csv across all subfolders.

    Args:
        root_folder (str): Path to the root folder to search

    Returns:
        dict: Dictionary with file numbers as keys and measurement data as values
    """
    # Pattern to match the CSV files
    file_pattern = r'DMA_Results_(\d{8}_\d{6})_truedyne_buffer\.csv'

    # Dictionary to store all measurements
    all_measurements = {}

    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            match = re.match(file_pattern, file)
            if match:
                # Extract the timestamp part as the identifier
                file_identifier = match.group(1)
                file_path = os.path.join(root, file)

                print(f"Processing file: {file_path}")

                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Check if required columns exist
                    required_columns = ['Timestamp', 'Scale Weight [g]', 'Scale Status']
                    missing_columns = [col for col in required_columns if col not in df.columns]

                    if missing_columns:
                        print(f"Warning: Missing columns {missing_columns} in file {file}")
                        continue

                    # Extract the required columns
                    extracted_data = {
                        'Timestamp': df['Timestamp'].tolist(),
                        'Scale Weight [g]': df['Scale Weight [g]'].tolist(),
                        'Scale Status': df['Scale Status'].tolist()
                    }

                    # Look for measurement description in the same folder or subfolders
                    description = extract_measurement_description(root)
                    if description:
                        extracted_data['Description'] = description

                    # Store in the main dictionary using the timestamp as key
                    all_measurements[file_identifier] = extracted_data

                    print(f"Successfully extracted {len(df)} records from {file}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue

    return all_measurements


def save_measurements_to_json(measurements, output_file='AllScaleMeasurements.json'):
    """
    Save the measurements dictionary to a JSON file.

    Args:
        measurements (dict): Dictionary containing all measurements
        output_file (str): Output JSON file name
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(measurements, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving to JSON file: {str(e)}")


def main():
    # Get the folder path from user input
    root_folder = input("Enter the root folder path to search: ").strip()

    # Validate the folder path
    if not os.path.exists(root_folder):
        print(f"Error: Folder '{root_folder}' does not exist.")
        return

    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a directory.")
        return

    print(f"Searching for CSV files in: {root_folder}")

    # Extract measurements from all matching CSV files
    measurements = extract_scale_measurements(root_folder)

    if not measurements:
        print("No matching CSV files found.")
        return

    print(f"Found and processed {len(measurements)} CSV files.")

    # Save to JSON file
    save_measurements_to_json(measurements)

    # Print summary
    print("\nSummary:")
    for file_id, data in measurements.items():
        desc_info = " (with description)" if 'Description' in data else ""
        print(f"File {file_id}: {len(data['Timestamp'])} records{desc_info}")


if __name__ == "__main__":
    main()