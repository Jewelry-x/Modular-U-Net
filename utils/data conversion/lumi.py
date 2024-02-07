import os
import numpy as np

# Replace 'your_directory' with the actual path to your directory containing RGB .npy files
directory_path = 'data/TTrain_label'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

# Iterate over each file
for file_name in file_list:
    # Construct the full path to the .npy file
    file_path = os.path.join(directory_path, file_name)

    # Load the RGB image from the .npy file
    rgb_image = np.load(file_path)

    # Convert RGB to grayscale using luminosity formula
    grayscale_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    # Save the grayscale image back to the original .npy file
    np.save(file_path, grayscale_image)
