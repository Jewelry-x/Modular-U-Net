import os

def rename_files_with_zeros(directory_path):
    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".npy"):
            # Extract the numeric part of the filename
            base_name, extension = os.path.splitext(filename)
            numeric_part = ''.join(filter(str.isdigit, base_name))

            # Add leading zeros to the numeric part
            padded_numeric_part = numeric_part.zfill(4)

            # Construct the new filename
            new_filename = f"frame_{padded_numeric_part}{extension}"

            # Rename the file
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            os.rename(old_path, new_path)

            print(f"Renamed: {filename} to {new_filename}")

if __name__ == "__main__":
    # Specify the directory containing .npy files
    directory_path = 'data\TTest_label'  # Replace 'your_directory' with the actual directory path

    # Rename files with leading zeros
    rename_files_with_zeros(directory_path)
