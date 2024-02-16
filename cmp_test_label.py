import matplotlib.pyplot as plt
import numpy as np
from model import UNet
import torch
import os
from config import *

# Check if the model file exists
if os.path.exists(os.path.join(RESULT_PATH, "saved_model_pt")):
    whole_model = torch.load(os.path.join(RESULT_PATH, "saved_model_pt"))
    # Construct the model details string
    model_details = ""
    if "data" in whole_model:
        model_details += "Data trained on: " + str(whole_model.get("data")) + "\n"
    if "learning_rate" in whole_model:
        model_details += (
            "Learning rate: " + str(whole_model.get("learning_rate")) + "\n"
        )
    if "pools" in whole_model:
        model_details += "Pooling layers used: " + str(whole_model.get("pools")) + "\n"
    if "reverse_pools" in whole_model:
        model_details += "Reverse pooling used: " + str(whole_model.get("reverse_pools")) + "\n"
    if "data_augmentations" in whole_model:
        model_details += (
            "Data Augmentations: " + str(whole_model.get("data_augmentations")) + "\n"
        )
    if "tested_on" in whole_model:
        for i, data in enumerate(whole_model["tested_on"]):
            model_details += f"{data} IOU: {whole_model['IOU'][i]}\n{data} DC: {whole_model['DC'][i]}\n"
    if "image_size" in whole_model:
        model_details += "Image Size: " + str(whole_model.get("image_size")) + "\n"
    if "optimizer" in whole_model:
        model_details += "Optimizer: " + str(whole_model.get("optimizer")) + "\n"
    if "epoch" in whole_model:
        model_details += "Epoch Saved: " + str(whole_model.get("epoch")) + "\n"
    if "early_stopping" in whole_model:
        model_details += (
            "Early Stopping: " + str(whole_model.get("early_stopping")) + "\n"
        )
    if "early_stopping_epochs" in whole_model:
        model_details += (
            "Early Stopping Epoch Limit: "
            + str(whole_model.get("early_stopping_epochs"))
            + "\n"
        )
else:
    model_details = "Model not found or loaded"


# Get the list of files in the directory
all_files = os.listdir(RESULT_PATH)

# Filter the files based on the extension
filtered_files = [file for file in all_files if file.endswith(".npy")]

# Create subplots for images and model details
fig, axs = plt.subplots(len(filtered_files), 4, gridspec_kw={"width_ratios": [1, 1, 1, 0.5]})

idx = 0
for file in filtered_files:
    data = file[:-15]
    created_mask = np.load(os.path.join(RESULT_PATH, file))
    original_mask = np.load(os.path.join(DATA_PATH, TESTING_DATA_MASK_LOCATION[TESTING_DATA.index(data)], MASK_DEFINITION % 0000))
    original_image = np.load(os.path.join(DATA_PATH, TESTING_DATA_LOCATION[TESTING_DATA.index(data)], IMAGE_DEFINITION % 0000) )

    # Plot Phantom images
    axs[idx, 0].imshow(np.squeeze(original_image), cmap="gray")
    axs[idx, 0].set_title(data + " Original Image")

    axs[idx, 1].imshow(np.squeeze(original_mask), cmap="gray")
    axs[idx, 1].set_title(data + " Mask")

    axs[idx, 2].imshow(np.squeeze(created_mask), cmap="gray")
    axs[idx, 2].set_title(data + " Model Created Mask")

    axs[idx, 3].axis("off")

    idx += 1

# Hide ticks and labels for the empty subplot
axs[0, 3].text(0, 0.5, model_details, fontsize=14, ha="left", va="center")

plt.show()