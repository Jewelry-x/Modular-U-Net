import matplotlib.pyplot as plt
import numpy as np

# Assuming data is a 2D NumPy array with shape (256, 256, 2)
data = np.load('data\TTrain\\frame_0000.npy')

# Visualize the data using different colormaps for each channel
print(np.max(data[:,:,0]))
plt.subplot(1, 1, 1)
plt.imshow(data[:,:,0], cmap='gray', vmin=0, vmax=np.max(data[:,:,0]))
plt.title('Channel 1')

plt.subplot(1, 2, 2)
plt.imshow(data[:,:,1], cmap='gray', vmin=0, vmax=np.max(data[:,:,1]))
plt.title('Channel 2')

plt.show()
