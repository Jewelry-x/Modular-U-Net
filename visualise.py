import matplotlib.pyplot as plt
import numpy as np

# Assuming data is a 2D NumPy array
data = np.load('data\TTrain\\frame_0000.npy')
# data = data / 255.0
# Visualize the data
print(np.max(data[:,:]))
plt.imshow(data, cmap='gray')  # Use an appropriate colormap and set vmin/vmax
plt.show()
