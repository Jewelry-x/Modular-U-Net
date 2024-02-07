import os

def rename_frames(directory, start_number):
    if not os.path.exists(directory):
        print(f"Error: Directory not found - {directory}")
        return

    counter = 0
    for filename in os.listdir(directory):
        if filename.startswith("frame_") and filename.endswith(".npy"):
            
            old_path = os.path.join(directory, filename)

            # Calculate the new frame number
            new_number = start_number + counter

            # Construct the new filename
            new_filename = f"frame_{new_number}.npy"
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")
            counter += 1

if __name__ == "__main__":
    directory_path = "data\TTest"  # Replace with your directory path
    starting_number = 0  # Replace with your desired starting number

    rename_frames(directory_path, starting_number)
