import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt


def lucas_kanade(img1, img2 ,patch_size=15, features=None):
    Gx = np.array([[-1, 0, 1]])
    Gy = Gx.T
    Gt = np.array([[-1, 1]])

    if features == None:
        h, w = img1.shape
        rows = np.arange(patch_size // 2, h - patch_size //2, patch_size)
        columns = np.arange(patch_size // 2, w - patch_size // 2, patch_size)
        mesh = np.meshgrid(rows, columns)
        center_points = list(zip(mesh[0].reshape(-1, ), mesh[1].reshape(-1, )))
    else:
        center_points = list(zip(features[0].reshape(-1, ), features[1].reshape(-1, )))

    v = np.zeros((len(center_points), 2, 1))

    for i, (r, c) in enumerate(center_points):
        img1_patch = img1[max(0, r - patch_size // 2):r + patch_size // 2 + 1, max(0, c - patch_size // 2):c + patch_size // 2 + 1]
        img2_patch = img2[max(0, r - patch_size // 2):r + patch_size // 2 + 1, max(0, c - patch_size // 2):c + patch_size // 2 + 1]

        A = np.zeros((img1_patch.size, 2))
        Ix = convolve2d(img1_patch, Gx, boundary="symm", mode="same")
        Iy = convolve2d(img1_patch, Gy, boundary="symm", mode="same")
        b = -convolve2d(np.vstack((np.ravel(img1_patch), np.ravel(img2_patch))).T, Gt, mode="valid")

        A[:, 0] = np.ravel(Ix)
        A[:, 1] = np.ravel(Iy)

        v[i] = np.linalg.inv(A.T @ A) @ A.T @ b

    return v


if __name__ == "__main__":
    patch_size = 15

    img = cv2.imread("./sphere1.ppm")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)

    img2 = cv2.imread("./sphere2.ppm")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img2 = img2.astype(np.float32)

    v = lucas_kanade(img, img2, patch_size)

    rows = np.arange(patch_size // 2, img.shape[0] - patch_size // 2, patch_size)
    columns = np.arange(patch_size // 2, img.shape[1] - patch_size // 2, patch_size)

    X, Y = np.meshgrid(np.arange(patch_size // 2, img.shape[0] - patch_size // 2, patch_size),
                       np.arange(patch_size // 2, img.shape[1] - patch_size // 2, patch_size))

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.quiver(X.T, Y.T, v[:, 0, 0], v[:, 1, 0], angles="xy", scale=20, color="r")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
