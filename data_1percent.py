import os
import numpy as np

def process_npy_files(input_folder, output_folder, train_subfolder='train', train_percentage=0.001, default_percentage=0.01):
    """
    Recursively processes all .npy files in a folder, using different percentages for specific subfolders.
    
    Args:
        input_folder (str): Path to the input folder containing .npy files.
        output_folder (str): Path to the output folder to save processed files.
        train_subfolder (str): Specific subfolder to use a different percentage.
        train_percentage (float): Percentage of data to retain for the specified subfolder.
        default_percentage (float): Default percentage of data to retain for other folders.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)

                # Determine the percentage based on the folder
                relative_path = os.path.relpath(root, input_folder)
                if relative_path.startswith(train_subfolder):
                    percentage = train_percentage
                else:
                    percentage = default_percentage

                # Create directories in the output folder if they don't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Load the .npy file
                data = np.load(input_path)
                reduced_data_size = int(len(data) * percentage)
                reduced_data = data[:reduced_data_size]

                # Save the reduced data to the output folder
                np.save(output_path, reduced_data)
                print(f"Processed: {input_path} -> {output_path} with {percentage*100:.2f}% of data")

# Input folder containing the .npy files
input_folder = './data/DL24FA'

# Output folder to save the processed .npy files
output_folder = './data/DL24FA_reduced'

# Process the .npy files
process_npy_files(input_folder, output_folder, train_subfolder='train', train_percentage=0.001, default_percentage=0.01)
