import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
file_path = 'data\TTrain\\frame_0000.npy'

# Load the NumPy array from the .npy file
image = np.load(file_path)

# Check if the array has 2 channels
if image.shape[-1] == 2:
    print(f"The array in '{file_path}' has 2 channels." + str(image.shape))
else:
    print(f"The array in '{file_path}' does not have 2 channels." + str(image.shape))
