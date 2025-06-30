#!/usr/bin/env python3
"""
Script to remove TrueDyne error lines from log files.
Filters out lines containing "Error reading TrueDyne" messages.
"""

import os
import re
from datetime import datetime


def filter_truedyne_errors(input_file, output_file=None):
    """
    Remove lines containing TrueDyne errors from a log file.

    Args:
        input_file (str): Path to the input log file
        output_file (str): Path to the output file (optional, defaults to input_file with _filtered suffix)

    Returns:
        tuple: (lines_removed, total_lines)
    """

    # Generate output filename if not provided
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_filtered{ext}"

    # Pattern to match TrueDyne error lines
    # Matches lines with timestamp followed by "Error reading TrueDyne"
    error_pattern = re.compile(r'^\[\d{2}:\d{2}:\d{2}\]\s+Error reading TrueDyne')

    lines_removed = 0
    total_lines = 0
    filtered_lines = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                # Check if line matches TrueDyne error pattern
                if error_pattern.match(line.strip()):
                    lines_removed += 1
                    print(f"Removing: {line.strip()}")
                else:
                    filtered_lines.append(line)

        # Write filtered content to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)

        print(f"\nFiltering complete!")
        print(f"Total lines processed: {total_lines}")
        print(f"Lines removed: {lines_removed}")
        print(f"Lines remaining: {total_lines - lines_removed}")
        print(f"Filtered file saved as: {output_file}")

        return lines_removed, total_lines

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return 0, 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0


def main():
    """Main function to run the script."""

    # You can modify these paths as needed
    input_file = "ErrorLog_cleaned.txt"
    print("TrueDyne Error Filter Script")
    print("=" * 40)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found.")
        input_file = input("Please enter the path to your log file: ").strip()

        if not os.path.exists(input_file):
            print("File not found. Exiting.")
            return

    # Ask user for output file preference
    use_custom_output = input("Do you want to specify an output filename? (y/n): ").lower().strip()

    if use_custom_output == 'y':
        output_file = input("Enter output filename: ").strip()
    else:
        output_file = None

    # Filter the file
    print(f"\nProcessing file: {input_file}")
    filter_truedyne_errors(input_file, output_file)


if __name__ == "__main__":
    main()



