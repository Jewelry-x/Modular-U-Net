import matplotlib.pyplot as plt
import numpy as np
from model import UNet
import torch
import os

P_label = "result\PTest_test_label.npy"
T_label = "result\TTest_test_label.npy"
model_path = "result\saved_model_pt"

# Check if the model file exists
if os.path.exists(model_path):
    whole_model = torch.load(model_path)
    # Construct the model details string
    model_details = ""
    if "data" in whole_model:
        model_details += "Data trained on: " + str(whole_model.get("data")) + "\n"
    if "learning_rate" in whole_model:
        model_details += (
            "Learning rate: " + str(whole_model.get("learning_rate")) + "\n"
        )
    if "phantom_IOU" in whole_model:
        model_details += "Phantom IoU: " + str(whole_model.get("phantom_IOU")) + "\n"
    if "phantom_DC" in whole_model:
        model_details += "Phantom DC: " + str(whole_model.get("phantom_DC")) + "\n"
    if "T1T6_IOU" in whole_model:
        model_details += "T1T6 IoU: " + str(whole_model.get("T1T6_IOU")) + "\n"
    if "T1T6_DC" in whole_model:
        model_details += "T1T6 DC: " + str(whole_model.get("T1T6_DC")) + "\n"
    if "pools" in whole_model:
        model_details += "Pooling layers used: " + str(whole_model.get("pools")) + "\n"
    if "data_augmentations" in whole_model:
        model_details += (
            "Data Augmentations: " + str(whole_model.get("data_augmentations")) + "\n"
        )
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

# Create subplots for images and model details
fig, axs = plt.subplots(2, 4, gridspec_kw={"width_ratios": [1, 1, 1, 0.5]})

# Check if P_label exists
if os.path.exists(P_label):
    Phantom_created = np.load(P_label)
    Pimage = np.load("data\\PTest\\frame_0000.npy")
    Pmask = np.load("data\\PTest_label\\frame_0000.npy")

    # Plot Phantom images
    axs[0, 0].imshow(np.squeeze(Pimage), cmap="gray")
    axs[0, 0].set_title("Phantom Original Image")

    axs[0, 1].imshow(np.squeeze(Pmask), cmap="gray")
    axs[0, 1].set_title("Phantom Mask")

    axs[0, 2].imshow(np.squeeze(Phantom_created), cmap="gray")
    axs[0, 2].set_title("Phantom Model Created Mask")

# Check if T_label exists
if os.path.exists(T_label):
    T1T6_created = np.load(T_label)
    Timage = np.load("data\\TTest\\frame_0000.npy")
    Tmask = np.load("data\\TTest_label\\frame_0000.npy")

    # Plot T1-T6 images
    axs[1, 0].imshow(np.squeeze(Timage), cmap="gray")
    axs[1, 0].set_title("T1-T6 Original Image")

    axs[1, 1].imshow(np.squeeze(Tmask), cmap="gray")
    axs[1, 1].set_title("T1-T6 Mask")

    axs[1, 2].imshow(np.squeeze(T1T6_created), cmap="gray")
    axs[1, 2].set_title("T1-T6 Model Created Mask")

# Hide ticks and labels for the empty subplot
axs[0, 3].axis("off")

# Adjust layout to add space for the text
plt.subplots_adjust(bottom=0.1, top=0.9)

# Add model details below the images
axs[0, 3].text(0, 0.5, model_details, fontsize=14, ha="left", va="center")
axs[1, 3].axis("off")

plt.show()
