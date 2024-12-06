import os
import numpy as np

def process_npy_files(input_folder, output_folder, percentage=0.001):
    """
    Recursively processes all .npy files in a folder, keeping only the first percentage of the data.
    
    Args:
        input_folder (str): Path to the input folder containing .npy files.
        output_folder (str): Path to the output folder to save processed files.
        percentage (float): Percentage of the data to retain (default is 0.1%).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)

                # Create directories in the output folder if they don't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Load the .npy file
                data = np.load(input_path)
                reduced_data_size = int(len(data) * percentage)
                reduced_data = data[:reduced_data_size]

                # Save the reduced data to the output folder
                np.save(output_path, reduced_data)
                print(f"Processed: {input_path} -> {output_path}")

# Input folder containing the .npy files
input_folder = './data/DL24FA'

# Output folder to save the processed .npy files
output_folder = './data/DL24FA_reduced'

# Process the .npy files
process_npy_files(input_folder, output_folder, percentage=0.001)
