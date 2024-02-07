import os
import numpy as np

def convert_float_to_int(file_path):
    # Load the .npy file into a NumPy array
    array_data = np.load(file_path)

    # Convert float values to integers
    array_data = array_data.astype(np.int32)

    # Save the modified array back to the same file
    np.save(file_path, array_data)

if __name__ == "__main__":
    # Specify the directory containing .npy files
    directory_path = 'data/PTest/'  # Replace 'your_directory' with the actual directory path

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory_path, filename)
            convert_float_to_int(file_path)
            print(f"Converted values to integers for: {filename}")
