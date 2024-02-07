import numpy as np

def get_image_properties(array):
    if len(array.shape) == 3:
        height, width, channels = array.shape
        return f"Image dimensions: {height}x{width}, Channels: {channels}"
    elif len(array.shape) == 2:
        height, width = array.shape
        return f"Grayscale image dimensions: {height}x{width}"
    else:
        return "Invalid image shape"

# Example usage:
file_path = 'data\PTrain\\frame_0002.npy'  # Replace 'your_file.npy' with the actual file path
array_data = np.load(file_path)

properties = get_image_properties(array_data)
print(properties)
