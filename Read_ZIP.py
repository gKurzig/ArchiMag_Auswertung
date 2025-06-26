import json
import csv
import zipfile
import os
from datetime import datetime


def process_zip_to_csv(zip_file_path, output_csv_path):
    """
    Process all JSON files in a zip archive and convert to CSV format.

    Args:
        zip_file_path (str): Path to the zip file containing JSON files
        output_csv_path (str): Path for the output CSV file
    """

    # CSV header as specified
    header = [
        "Sample Number",
        "Date of Measurement",
        "Sample Name",
        "Product Name",
        "Status",
        "T (cell)",
        "Density"
    ]

    rows = []

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Get all JSON files in the zip
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]

            for json_file in json_files:
                try:
                    # Read JSON file from zip
                    with zip_ref.open(json_file) as file:
                        data = json.load(file)

                    # Extract data from JSON structure
                    dma4501 = data.get('dma4501', {})

                    # Extract required fields
                    sample_number = dma4501.get('measurement_number', '')
                    date_measurement = dma4501.get('date', '')
                    sample_name = dma4501.get('description', '')
                    product_name = dma4501.get('category', '')
                    status = dma4501.get('measurement_point', '0')  # Default to 0 as shown in sample
                    temperature = dma4501.get('temperature', '')
                    density = dma4501.get('density', '')

                    # Format temperature to 2 decimal places if it's a number
                    try:
                        if temperature:
                            temperature = f"{float(temperature):.2f}"
                    except (ValueError, TypeError):
                        temperature = str(temperature)

                    # Format density to match the sample format (5 decimal places)
                    try:
                        if density:
                            density = f"{float(density):.5f}"
                    except (ValueError, TypeError):
                        density = str(density)

                    # Create row
                    row = [
                        sample_number,
                        date_measurement,
                        sample_name,
                        product_name,
                        status,
                        temperature,
                        density
                    ]

                    rows.append(row)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON file {json_file}: {e}")
                except Exception as e:
                    print(f"Error processing file {json_file}: {e}")

        # Sort rows by sample number if possible
        try:
            rows.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        except:
            pass  # If sorting fails, keep original order

        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(header)
            writer.writerows(rows)

        print(f"Successfully processed {len(rows)} records from {len(json_files)} JSON files.")
        print(f"CSV file saved as: {output_csv_path}")

    except zipfile.BadZipFile:
        print(f"Error: {zip_file_path} is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: Zip file {zip_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_directory_to_csv(directory_path, output_csv_path):
    """
    Process all JSON files in a directory and convert to CSV format.

    Args:
        directory_path (str): Path to the directory containing JSON files
        output_csv_path (str): Path for the output CSV file
    """

    # CSV header as specified
    header = [
        "Sample Number",
        "Date of Measurement",
        "Sample Name",
        "Product Name",
        "Status",
        "T (cell)",
        "Density"
    ]

    rows = []

    try:
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

        if not json_files:
            print(f"No JSON files found in directory: {directory_path}")
            return

        for json_file in json_files:
            try:
                file_path = os.path.join(directory_path, json_file)

                # Read JSON file
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Extract data from JSON structure
                dma4501 = data.get('dma4501', {})

                # Extract required fields
                sample_number = dma4501.get('measurement_number', '')
                date_measurement = dma4501.get('date', '')
                sample_name = dma4501.get('description', '')
                product_name = dma4501.get('category', '')
                status = dma4501.get('measurement_point', '0')  # Default to 0 as shown in sample
                temperature = dma4501.get('temperature', '')
                density = dma4501.get('density', '')

                # Format temperature to 2 decimal places if it's a number
                try:
                    if temperature:
                        temperature = f"{float(temperature):.2f}"
                except (ValueError, TypeError):
                    temperature = str(temperature)

                # Format density to match the sample format (5 decimal places)
                try:
                    if density:
                        density = f"{float(density):.5f}"
                except (ValueError, TypeError):
                    density = str(density)

                # Create row
                row = [
                    sample_number,
                    date_measurement,
                    sample_name,
                    product_name,
                    status,
                    temperature,
                    density
                ]

                rows.append(row)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {json_file}: {e}")
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")

        # Sort rows by sample number if possible
        try:
            rows.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        except:
            pass  # If sorting fails, keep original order

        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(header)
            writer.writerows(rows)

        print(f"Successfully processed {len(rows)} records from {len(json_files)} JSON files.")
        print(f"CSV file saved as: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Directory {directory_path} not found.")
    except PermissionError:
        print(f"Error: Permission denied accessing {directory_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to run the converter.
    Modify the file paths below as needed.
    """

    # Input path - can be either a zip file or a directory
    input_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250624_Messreihe_6\measurements_20250624_173820.zip"

    # Output CSV file path - change this to your desired output location
    #output_csv_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250623_Messreihe_5\measurement_data.csv"
    output_csv_path = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250624_Messreihe_6\measurements_20250624_173820.csv"
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        return

    # Check if it's a directory or zip file
    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        process_directory_to_csv(input_path, output_csv_path)
    elif input_path.endswith('.zip'):
        print(f"Processing zip file: {input_path}")
        process_zip_to_csv(input_path, output_csv_path)
    else:
        print(f"Error: Input path must be either a directory containing JSON files or a zip file.")
        print(f"Current path: {input_path}")


def process_measurement_data(input_path, output_csv_path=None):
    """
    External function to process measurement data and return both the generated CSV and truedyne buffer file.

    Args:
        input_path (str): Path to directory or zip file containing JSON files
        output_csv_path (str, optional): Path for output CSV. If None, uses input directory + 'measurement_data.csv'

    Returns:
        tuple: (generated_csv_path, truedyne_buffer_path)
               Both paths will be None if files are not found or errors occur
    """

    try:
        # Generate output CSV path if not provided
        if output_csv_path is None:
            if os.path.isdir(input_path):
                output_csv_path = os.path.join(input_path, "measurement_data.csv")
            else:
                # For zip files, use the same directory as the zip file
                dir_path = os.path.dirname(input_path)
                output_csv_path = os.path.join(dir_path, "measurement_data.csv")

        truedyne_buffer_path = None

        # Check if input path exists
        if not os.path.exists(input_path):
            print(f"Error: Path does not exist: {input_path}")
            return None, None

        # Process JSON files and find truedyne buffer file
        if os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            process_directory_to_csv(input_path, output_csv_path)

            # Look for truedyne buffer CSV file in directory
            csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv') and 'truedyne_buffer' in f.lower()]
            if csv_files:
                truedyne_buffer_path = os.path.join(input_path, csv_files[0])
                print(f"Found truedyne buffer file: {csv_files[0]}")
            else:
                print("No truedyne buffer CSV file found in directory")

        elif input_path.endswith('.zip'):
            print(f"Processing zip file: {input_path}")
            process_zip_to_csv(input_path, output_csv_path)

            # Look for truedyne buffer CSV file in zip and extract it
            try:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv') and 'truedyne_buffer' in f.lower()]
                    if csv_files:
                        # Extract the truedyne buffer file to the same directory as output CSV
                        buffer_filename = os.path.basename(csv_files[0])
                        extract_dir = os.path.dirname(output_csv_path)
                        truedyne_buffer_path = os.path.join(extract_dir, buffer_filename)

                        # Extract the file
                        with zip_ref.open(csv_files[0]) as source, open(truedyne_buffer_path, 'wb') as target:
                            target.write(source.read())

                        print(f"Extracted truedyne buffer file: {buffer_filename}")
                    else:
                        print("No truedyne buffer CSV file found in zip")
            except Exception as e:
                print(f"Error extracting truedyne buffer file: {e}")
        else:
            print(f"Error: Input path must be either a directory containing JSON files or a zip file.")
            return None, None

        # Verify that the generated CSV file exists
        if os.path.exists(output_csv_path):
            print(f"Successfully created CSV: {output_csv_path}")
            return output_csv_path, truedyne_buffer_path
        else:
            print("Error: CSV file was not created successfully")
            return None, truedyne_buffer_path

    except Exception as e:
        print(f"An error occurred in process_measurement_data: {e}")
        return None, None


if __name__ == "__main__":
    main()