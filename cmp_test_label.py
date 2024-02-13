import matplotlib.pyplot as plt
import numpy as np
import os

P_label = "result\PTest_test_label.npy"
T_label = "result\TTest_test_label.npy"


if os.path.exists(P_label):
    Phantom_created = np.load(P_label)
    Pimage = np.load("data\\PTest\\frame_0000.npy")
    Pmask = np.load("data\\PTest_label\\frame_0000.npy")

if os.path.exists(T_label):
    T1T6_created = np.load(T_label)
    Timage = np.load("data\\TTest\\frame_0000.npy")
    Tmask = np.load("data\\TTest_label\\frame_0000.npy")


if os.path.exists(P_label):
    plt.subplot(2, 3, 1)
    plt.imshow(np.squeeze(Pimage), cmap="gray")
    plt.title("Phantom Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(np.squeeze(Pmask), cmap="gray")
    plt.title("Phantom Mask")

    plt.subplot(2, 3, 3)
    plt.imshow(np.squeeze(Phantom_created), cmap="gray")
    plt.title("Phantom Model Created Mask")

if os.path.exists(T_label):
    plt.subplot(2, 3, 4)
    plt.imshow(np.squeeze(Timage), cmap="gray")
    plt.title("T1-T6 Original Image")

    plt.subplot(2, 3, 5)
    plt.imshow(np.squeeze(Tmask), cmap="gray")
    plt.title("T1-T6 Mask")

    plt.subplot(2, 3, 6)
    plt.imshow(np.squeeze(T1T6_created), cmap="gray")
    plt.title("T1T6 Model Created Mask")

plt.show()
