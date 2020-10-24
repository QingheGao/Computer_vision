import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_formation(albedo_image, shading_image, original_image):
    albedo_image = albedo_image.astype(np.float32)
    shading_image = shading_image.astype(np.float32)
    reconstructed_image = (albedo_image / 255) * (shading_image / 255)

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(reconstructed_image)
    axs[0, 1].set_title(r'Reconstructed')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(albedo_image.astype(np.uint8))
    axs[1, 0].set_title(r'Albedo')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(shading_image.astype(np.uint8))
    axs[1, 1].set_title(r'Shading')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    plt.show()


if __name__ == '__main__':
    original = cv2.imread('ball.png')[:, :, ::-1]
    albedo = cv2.imread('ball_albedo.png')[:, :, ::-1]
    shading = cv2.imread('ball_shading.png')[:, :, ::-1]

    image_formation(albedo, shading, original)
