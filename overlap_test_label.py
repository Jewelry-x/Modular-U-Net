import matplotlib.pyplot as plt
import numpy as np
import os

P_label = "result\PTest_test_label.npy"
T_label = "result\TTest_test_label.npy"

# Create subplots for images
fig, axs = plt.subplots(2, 3)

# Check if P_label exists
if os.path.exists(P_label):
    Phantom_created = np.load(P_label)
    Pimage = np.load("data\\PTest\\frame_0000.npy")
    Pmask = np.load("data\\PTest_label\\frame_0000.npy")

    # Plot Phantom images with overlap
    axs[0, 0].imshow(np.squeeze(Pimage), cmap="gray")
    axs[0, 0].imshow(np.squeeze(Pmask), cmap="jet", alpha=0.5)
    axs[0, 0].set_title("Original Image with Mask")

    axs[0, 1].imshow(np.squeeze(Pimage), cmap="gray")
    axs[0, 1].imshow(np.squeeze(Phantom_created), cmap="jet", alpha=0.5)
    axs[0, 1].set_title("Original Image with Created Mask")

    axs[0, 2].imshow(np.squeeze(Pmask), cmap="gray")
    axs[0, 2].imshow(np.squeeze(Phantom_created), cmap="jet", alpha=0.5)
    axs[0, 2].set_title("Original and Created Mask Overlap")

# Check if T_label exists
if os.path.exists(T_label):
    T1T6_created = np.load(T_label)
    Timage = np.load("data\\TTest\\frame_0000.npy")
    Tmask = np.load("data\\TTest_label\\frame_0000.npy")

    # Plot T1-T6 images with overlap
    axs[1, 0].imshow(np.squeeze(Timage), cmap="gray")
    axs[1, 0].imshow(np.squeeze(Tmask), cmap="jet", alpha=0.5)
    axs[1, 0].set_title("Original Image with Mask")

    axs[1, 1].imshow(np.squeeze(Timage), cmap="gray")
    axs[1, 1].imshow(np.squeeze(T1T6_created), cmap="jet", alpha=0.5)
    axs[1, 1].set_title("Original Image with Created Mask")

    axs[1, 2].imshow(np.squeeze(Tmask), cmap="gray")
    axs[1, 2].imshow(np.squeeze(T1T6_created), cmap="jet", alpha=0.5)
    axs[1, 2].set_title("Original and Created Mask Overlap")

plt.show()
