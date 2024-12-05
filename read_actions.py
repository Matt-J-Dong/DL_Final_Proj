import numpy as np

# Define the file path relative to the current directory
file_path = '/scratch/DL24FA/train/actions.npy'

try:
    # Load the .npy file
    data = np.load(file_path)
    
    # Print the first few elements of the file
    print("Head of the contents in actions.npy:")
    print(data[:10])  # Adjust the number of elements to print as needed
    
except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
