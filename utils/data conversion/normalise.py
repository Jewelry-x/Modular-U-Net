import os
import cv2
import numpy as np

def normalize_frames(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found - {directory_path}")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(directory_path, filename)

            # Read the image
            image = cv2.imread(file_path)

            if image is None:
                print(f"Error: Unable to read image - {file_path}")
                continue

            # Normalize pixel values to the range [0, 1]
            normalized_image = image.astype(np.float32) / 255.0

            # Save the normalized image back to the directory
            cv2.imwrite(file_path, (normalized_image * 255).astype(np.uint8))

            print(f"Normalized and saved {filename}")

if __name__ == "__main__":
    frames_directory = "data/TTest"  # Replace with your frames directory path

    normalize_frames(frames_directory)
