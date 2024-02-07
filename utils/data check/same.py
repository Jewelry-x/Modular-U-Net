import numpy as np

def load_npy(file_path):
    return np.load(file_path)

def check_identical_frames(file_path):
    # Load the .npy file
    frames = load_npy(file_path)

    # Check if all frames are identical
    if np.all(frames[0] == frames[1:]):
        print("All frames in the .npy file are identical.")
    else:
        print("Frames in the .npy file are not identical.")

# Example usage:
file_path = 'data\TTrain\\frame_0000.npy'  # Replace with the actual path to the .npy file
check_identical_frames(file_path)
