import matplotlib.pyplot as plt
import numpy as np

# Set frame and data to show image and mask
frame = "0000"
data = "TTrain"

image = np.load("data\\" + data + "\\frame_" + str(frame) + ".npy")
mask = np.load("data\\" + data + "_label\\frame_" + str(frame) + ".npy")

plt.figure(figsize=(10, 5))

# Plot the image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Image")

# Plot the mask
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.show()
