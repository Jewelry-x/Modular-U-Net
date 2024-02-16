import matplotlib.pyplot as plt
import numpy as np
import os
from config import * 

# Get the list of files in the directory
all_files = os.listdir(RESULT_PATH)

# Filter the files based on the extension
filtered_files = [file for file in all_files if file.endswith(".npy")]

# Create subplots for images and model details
fig, axs = plt.subplots(len(filtered_files), 3)

idx = 0
for file in filtered_files:
    data = file[:-15]
    created_mask = np.load(os.path.join(RESULT_PATH, file))
    original_mask = np.load(os.path.join(DATA_PATH, TESTING_DATA_MASK_LOCATION[TESTING_DATA.index(data)], MASK_DEFINITION % 0000))
    original_image = np.load(os.path.join(DATA_PATH, TESTING_DATA_LOCATION[TESTING_DATA.index(data)], IMAGE_DEFINITION % 0000) )

 
    axs[idx, 0].imshow(np.squeeze(original_image), cmap="gray")
    axs[idx, 0].set_title(data + " Original Image with Mask")
    axs[idx, 0].imshow(np.squeeze(original_mask), cmap="jet", alpha=0.5)

    axs[idx, 1].imshow(np.squeeze(original_image), cmap="gray")
    axs[idx, 1].imshow(np.squeeze(created_mask), cmap="jet", alpha=0.5)
    axs[idx, 1].set_title(data + " Original Image with Created Mask")

    axs[idx, 2].imshow(np.squeeze(original_mask), cmap="gray")
    axs[idx, 2].imshow(np.squeeze(created_mask), cmap="jet", alpha=0.5)
    axs[idx, 2].set_title(data + " Original and Created Mask Overlap")

    idx += 1

plt.show()