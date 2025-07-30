#!/usr/bin/env python3
"""
Script to filter out TrueDyne sensor error lines from log files.
Removes lines containing "Error reading TrueDyne DGFI1" or "Error reading TrueDyne MGCE1"
that appear between "Next measurement scheduled" and "Starting measurement" markers.
"""

import re
import sys
from pathlib import Path


def filter_log_file(input_file, output_file=None):
    """
    Filter out TrueDyne sensor error lines from log file.

    Args:
        input_file (str): Path to input log file
        output_file (str, optional): Path to output file. If None, overwrites input file.
    """

    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    filtered_lines = []
    in_measurement_gap = False

    # Regex patterns
    next_measurement_pattern = re.compile(r'Next measurement scheduled in \d+\.\d+ minutes')
    starting_measurement_pattern = re.compile(r'Starting measurement #\d+')
    error_pattern = re.compile(r'Error reading TrueDyne (DGFI1|MGCE1) sensor:')

    for line in lines:
        line_content = line.strip()

        # Check if we're entering a measurement gap
        if next_measurement_pattern.search(line_content):
            in_measurement_gap = True
            filtered_lines.append(line)
            continue

        # Check if we're exiting a measurement gap
        if starting_measurement_pattern.search(line_content):
            in_measurement_gap = False
            filtered_lines.append(line)
            continue

        # If we're in a measurement gap and this is a TrueDyne error, skip it
        if in_measurement_gap and error_pattern.search(line_content):
            continue

        # Keep all other lines
        filtered_lines.append(line)

    # Write the filtered content
    output_path = output_file if output_file else input_file

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)

        original_count = len(lines)
        filtered_count = len(filtered_lines)
        removed_count = original_count - filtered_count

        print(f"Filtering complete:")
        print(f"  Original lines: {original_count}")
        print(f"  Filtered lines: {filtered_count}")
        print(f"  Removed lines: {removed_count}")
        print(f"  Output saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error writing file: {e}")
        return False


def main():
    """Main function with hardcoded file paths."""

    # Hardcoded file paths - modify these as needed
    input_file = r"C:\Users\kurzm\Documents\Physik\MasterArbeit\Ergebnisse\20250721_Messreihe_15\measurements_20250721_151542\measurements_20250721_151542\gui_log_20250721_151525.txt"
    output_file = input_file.rsplit('.', 1)[0] + "_cleaned.txt"

    # Validate input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Run the filter
    success = filter_log_file(input_file, output_file)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()