import numpy as np
import cv2
import matplotlib.pyplot as plt


def recoloring(albedo_image, shading_image, original_image):
    albedo_image[:, :, 0] = 0
    albedo_image[:, :, 2] = 0
    albedo_image[albedo_image == 141] = 255
    albedo_image = albedo_image.astype(np.float32)
    albedo_image = albedo_image.astype(np.float32)

    reconstructed_image = (albedo_image / 255) * (shading_image / 255)

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))

    axs[0].imshow(original_image)
    axs[0].set_title('Original')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(reconstructed_image)
    axs[1].set_title(r'Recolored')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.show()


if __name__ == '__main__':
    original = cv2.imread('ball.png')[:, :, ::-1]
    albedo = cv2.imread('ball_albedo.png')[:, :, ::-1]
    shading = cv2.imread('ball_shading.png')[:, :, ::-1]

    recoloring(albedo, shading, original)