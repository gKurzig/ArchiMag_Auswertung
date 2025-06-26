import tkinter as tk
from tkinter import filedialog
import os
import zipfile
import json
import csv


def select_zip_file():
    """
    Opens a file dialog to select a zip file and returns the path.

    Returns:
        str: Path to the selected zip file, or None if no file was selected
    """

    # Create a root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Configure file dialog options
    file_types = [
        ("ZIP files", "*.zip"),
        ("All files", "*.*")
    ]

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a ZIP file",
        filetypes=file_types,
        initialdir=os.getcwd()  # Start in current working directory
    )

    # Clean up the root window
    root.destroy()

    # Return the selected file path (will be empty string if cancelled, convert to None)
    return file_path if file_path else None


def process_zip_to_csv(zip_file_path, output_csv_path):
    """
    Process all JSON files in a zip archive and convert to CSV format.

    Args:
        zip_file_path (str): Path to the zip file containing JSON files
        output_csv_path (str): Path for the output CSV file

    Returns:
        bool: True if successful, False otherwise
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
        return True

    except Exception as e:
        print(f"Error processing zip file: {e}")
        return False


def fix_timestamp_timezone_issues(csv_path):
    """
    Fix timezone-aware timestamp parsing issues in CSV files.

    Args:
        csv_path (str): Path to CSV file with timestamp issues

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pandas as pd

        # Read the CSV
        df = pd.read_csv(csv_path, delimiter=';')

        # Fix the 'Date of Measurement' column if it exists
        if 'Date of Measurement' in df.columns:
            print("Fixing timezone issues in 'Date of Measurement' column...")

            # Convert to datetime, handling timezone-aware strings
            df['Date of Measurement'] = pd.to_datetime(df['Date of Measurement'], utc=True)

            # Save back to CSV
            df.to_csv(csv_path, index=False, sep=';')
            print(f"Fixed timestamp issues in: {csv_path}")
            return True
        else:
            print("No 'Date of Measurement' column found - no fixes needed")
            return True

    except Exception as e:
        print(f"Error fixing timestamp issues: {e}")
        return False


def setup_measurement_config(zip_file_path):
    """
    Set up configuration based on selected zip file.
    Creates folder, extracts files, and returns configuration variables.

    Args:
        zip_file_path (str): Path to the selected zip file

    Returns:
        dict: Configuration dictionary with all required paths and filenames
    """

    try:
        # Get zip file name without extension
        zip_name = os.path.splitext(os.path.basename(zip_file_path))[0]

        # Create data folder with same name as zip file
        zip_dir = os.path.dirname(zip_file_path)
        data_folder = os.path.join(zip_dir, zip_name)

        # Create directory if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)
        print(f"Created/using data folder: {data_folder}")

        # Generate DMA CSV file from JSON files in zip
        dma_file = "dma_messdaten.csv"
        dma_path = os.path.join(data_folder, dma_file)

        print("Processing JSON files to create DMA CSV...")
        if not process_zip_to_csv(zip_file_path, dma_path):
            print("Failed to process JSON files")
            return None

        # Fix timestamp timezone issues in the generated DMA CSV
        fix_timestamp_timezone_issues(dma_path)

        # Extract truedyne buffer CSV file from zip
        truedyne_file = None
        truedyne_path = None

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Look for truedyne buffer CSV file
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv') and 'truedyne_buffer' in f.lower()]

                if csv_files:
                    # Extract the truedyne buffer file
                    truedyne_file = os.path.basename(csv_files[0])
                    truedyne_path = os.path.join(data_folder, truedyne_file)

                    # Ensure the file doesn't exist or is writable
                    if os.path.exists(truedyne_path):
                        try:
                            os.chmod(truedyne_path, 0o666)  # Make writable
                        except:
                            pass

                    with zip_ref.open(csv_files[0]) as source, open(truedyne_path, 'wb') as target:
                        target.write(source.read())

                    print(f"Extracted truedyne buffer file: {truedyne_file}")
                else:
                    print("Warning: No truedyne buffer CSV file found in zip")

        except Exception as e:
            print(f"Error extracting truedyne buffer file: {e}")

            # Fallback: Try to find the buffer file in the same directory as the zip
            zip_dir = os.path.dirname(zip_file_path)
            possible_buffer_files = [f for f in os.listdir(zip_dir) if
                                     f.endswith('.csv') and 'truedyne_buffer' in f.lower()]

            if possible_buffer_files:
                print(f"Found potential buffer file in zip directory: {possible_buffer_files[0]}")
                source_buffer = os.path.join(zip_dir, possible_buffer_files[0])
                truedyne_file = possible_buffer_files[0]
                truedyne_path = os.path.join(data_folder, truedyne_file)

                try:
                    import shutil
                    shutil.copy2(source_buffer, truedyne_path)
                    print(f"Copied buffer file from zip directory: {truedyne_file}")
                except Exception as copy_error:
                    print(f"Failed to copy buffer file: {copy_error}")
                    truedyne_file = None
                    truedyne_path = None
            else:
                print("No buffer file found in zip directory either")
                truedyne_file = None
                truedyne_path = None

        # Standardize buffer CSV columns if truedyne file exists
        if truedyne_path and os.path.exists(truedyne_path):
            print("Standardizing buffer CSV columns...")

            # Try to standardize in place first
            standardized_path = standardize_buffer_columns(truedyne_path)

            if not standardized_path:
                # If that fails, try creating a new standardized file with a different name
                print("Creating standardized copy with different filename...")
                alt_filename = "truedyne_buffer_standardized.csv"
                alt_path = os.path.join(data_folder, alt_filename)
                standardized_path = standardize_buffer_columns(truedyne_path, alt_path)

                if standardized_path:
                    # Update the config to use the standardized file
                    truedyne_file = alt_filename
                    truedyne_path = alt_path
                    print(f"Using standardized buffer file: {alt_filename}")
                else:
                    print("Warning: Failed to create standardized buffer CSV")
        elif truedyne_path is None:
            # Try to read buffer file directly from zip and create standardized version
            print("Attempting to read buffer file directly from zip...")
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv') and 'truedyne_buffer' in f.lower()]

                    if csv_files:
                        import tempfile
                        import pandas as pd

                        # Read buffer file directly from zip into memory
                        with zip_ref.open(csv_files[0]) as source:
                            buffer_content = source.read()

                        # Write to temporary file first
                        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
                            temp_file.write(buffer_content)
                            temp_path = temp_file.name

                        # Create standardized version in our data folder
                        truedyne_file = "truedyne_buffer_standardized.csv"
                        truedyne_path = os.path.join(data_folder, truedyne_file)

                        standardized_path = standardize_buffer_columns(temp_path, truedyne_path)

                        # Clean up temp file
                        os.unlink(temp_path)

                        if standardized_path:
                            print(f"Successfully created standardized buffer from zip: {truedyne_file}")
                        else:
                            print("Failed to create standardized buffer from zip")
                            truedyne_file = None
                            truedyne_path = None

            except Exception as zip_error:
                print(f"Error reading buffer from zip: {zip_error}")
                truedyne_file = None
                truedyne_path = None

        # Set up configuration
        config = {
            'DATA_FOLDER': data_folder,
            'DMA_FILE': dma_file,
            'TRUEDYNE_FILE': truedyne_file,
            'OUTPUT_FILE': "fused_data.csv",
            'DMA_PATH': dma_path,
            'TRUEDYNE_PATH': truedyne_path,
            'OUTPUT_PATH': os.path.join(data_folder, "fused_data.csv")
        }

        return config

    except Exception as e:
        print(f"Error setting up configuration: {e}")
        return None


def standardize_buffer_columns(buffer_csv_path, output_path=None):
    """
    Opens the buffer CSV file and renames columns to match the critical_columns format.

    Args:
        buffer_csv_path (str): Path to the truedyne buffer CSV file
        output_path (str, optional): Path to save the standardized CSV. If None, overwrites original.

    Returns:
        str: Path to the standardized CSV file, or None if failed
    """

    if not buffer_csv_path or not os.path.exists(buffer_csv_path):
        print(f"Buffer CSV file not found: {buffer_csv_path}")
        return None

    # Target column names
    critical_columns = [
        'TrueDyne MGCE1 Density [kg/m**3]',
        'TrueDyne MGCE1 Press [Pa]',
        'TrueDyne MGCE1 Temp [C]',
        'TrueDyne DGFI1 Density [kg/m**3]',
        'TrueDyne DGFI1 Press [Pa]',
        'TrueDyne DGFI1 Temp [C]'
    ]

    try:
        # Import pandas here to avoid scope issues
        import pandas as pd

        # Try different encodings for the CSV file
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(buffer_csv_path, encoding=encoding)
                print(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print("Error: Could not read CSV file with any supported encoding")
            return None

        # Print current column names for debugging
        print("Current columns in buffer CSV:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        # Create column mapping based on common patterns
        column_mapping = {}

        # Map columns based on keywords and patterns
        for col in df.columns:
            col_lower = col.lower().strip()

            # MGCE1 mappings
            if 'mgce1' in col_lower or 'mgce' in col_lower:
                if 'density' in col_lower or 'dens' in col_lower:
                    column_mapping[col] = 'TrueDyne MGCE1 Density [kg/m**3]'
                elif 'pressure' in col_lower or 'press' in col_lower:
                    column_mapping[col] = 'TrueDyne MGCE1 Press [Pa]'
                elif 'temperature' in col_lower or 'temp' in col_lower:
                    column_mapping[col] = 'TrueDyne MGCE1 Temp [C]'

            # DGFI1 mappings
            elif 'dgfi1' in col_lower or 'dgfi' in col_lower:
                if 'density' in col_lower or 'dens' in col_lower:
                    column_mapping[col] = 'TrueDyne DGFI1 Density [kg/m**3]'
                elif 'pressure' in col_lower or 'press' in col_lower:
                    column_mapping[col] = 'TrueDyne DGFI1 Press [Pa]'
                elif 'temperature' in col_lower or 'temp' in col_lower:
                    column_mapping[col] = 'TrueDyne DGFI1 Temp [C]'

            # Alternative patterns - sometimes columns might have different naming
            elif 'density' in col_lower and '1' in col:
                if any(x in col_lower for x in ['mgce', 'gas']):
                    column_mapping[col] = 'TrueDyne MGCE1 Density [kg/m**3]'
                elif any(x in col_lower for x in ['dgfi', 'liquid']):
                    column_mapping[col] = 'TrueDyne DGFI1 Density [kg/m**3]'
            elif 'pressure' in col_lower and '1' in col:
                if any(x in col_lower for x in ['mgce', 'gas']):
                    column_mapping[col] = 'TrueDyne MGCE1 Press [Pa]'
                elif any(x in col_lower for x in ['dgfi', 'liquid']):
                    column_mapping[col] = 'TrueDyne DGFI1 Press [Pa]'
            elif 'temp' in col_lower and '1' in col:
                if any(x in col_lower for x in ['mgce', 'gas']):
                    column_mapping[col] = 'TrueDyne MGCE1 Temp [C]'
                elif any(x in col_lower for x in ['dgfi', 'liquid']):
                    column_mapping[col] = 'TrueDyne DGFI1 Temp [C]'

        print("\nColumn mapping:")
        for old_col, new_col in column_mapping.items():
            print(f"  '{old_col}' → '{new_col}'")

        # Apply the column mapping
        df_renamed = df.rename(columns=column_mapping)

        # Set output path
        if output_path is None:
            output_path = buffer_csv_path  # Overwrite original

        # Save the standardized CSV with the same encoding that worked for reading
        df_renamed.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\nStandardized buffer CSV saved: {output_path}")

        # Report which critical columns are available
        available_critical = [col for col in critical_columns if col in df_renamed.columns]
        missing_critical = [col for col in critical_columns if col not in df_renamed.columns]

        print(f"\nAvailable critical columns ({len(available_critical)}/{len(critical_columns)}):")
        for col in available_critical:
            print(f"  ✓ {col}")

        if missing_critical:
            print(f"\nMissing critical columns:")
            for col in missing_critical:
                print(f"  ✗ {col}")

            # If we're missing columns, let's see what columns we do have
            print(f"\nAll available columns in standardized file:")
            for col in df_renamed.columns:
                print(f"  - {col}")

        return output_path

    except ImportError:
        print("Error: pandas is required for this function. Install with: pip install pandas")
        return None
    except Exception as e:
        print(f"Error standardizing buffer CSV: {e}")
        return None


def main():
    """
    Main function to select zip file and set up configuration.

    Returns:
        dict: Configuration dictionary or None if failed
    """

    print("Select a ZIP file containing measurement data...")

    # Let user select zip file
    zip_path = select_zip_file()

    if not zip_path:
        print("No file selected.")
        return None

    print(f"Selected: {zip_path}")

    # Set up configuration
    config = setup_measurement_config(zip_path)

    if config:
        print("\n" + "=" * 50)
        print("CONFIGURATION SETUP COMPLETE")
        print("=" * 50)
        print(f"DATA_FOLDER = r\"{config['DATA_FOLDER']}\"")
        print(f"DMA_FILE = \"{config['DMA_FILE']}\"")
        print(f"TRUEDYNE_FILE = \"{config['TRUEDYNE_FILE']}\"")
        print(f"OUTPUT_FILE = \"{config['OUTPUT_FILE']}\"")
        print("=" * 50)

        return config
    else:
        print("Failed to set up configuration.")
        return None


if __name__ == "__main__":
    result = main()