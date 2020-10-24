import numpy as np
import cv2
import matplotlib.pyplot as plt


def colour_correction(input_image):
    new_image = input_image.astype(np.float32)

    gray = np.mean(new_image[:, :, 0]) + np.mean(new_image[:, :, 1]) + np.mean(new_image[:, :, 2])
    new_image[:, :, 0] *= gray/np.mean(new_image[:, :, 0])
    new_image[:, :, 1] *= gray/np.mean(new_image[:, :, 1])
    new_image[:, :, 2] *= gray/np.mean(new_image[:, :, 2])

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].imshow(input_image)
    axs[0].set_title('Original')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(new_image/new_image.max())
    axs[1].set_title('Color corrected')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.show()


if __name__ == '__main__':
    img_path = 'awb.jpg'
    I = cv2.imread(img_path)[:, :, ::-1]

    out_img = colour_correction(I)
