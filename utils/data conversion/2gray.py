import os
import numpy as np

# Replace 'your_directory' with the actual path to your directory containing .npy files
directory_path = 'data/TTest'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

# Iterate over each file
for file_name in file_list:
    # Construct the full path to the .npy file
    file_path = os.path.join(directory_path, file_name)

    # Load the NumPy array from the .npy file
    image = np.load(file_path)

    # Keep only the first channel (change the index as needed)
    single_channel_image = image[:, :, 0]

    # Now you can visualize or process the single-channel image
    # Add your processing or visualization code here

    # Save the modified image back to the original .npy file
    np.save(file_path, single_channel_image)
