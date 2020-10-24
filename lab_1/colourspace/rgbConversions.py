import numpy as np
import cv2

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods


    # stack all images into one tensor
    new_image = np.zeros(input_image.shape[:2] + (4,))
    # ligtness method
    new_image[:, :, 0] = (input_image.max(axis=2) + input_image.min(axis=2)) / 2

    # average method
    new_image[:, :, 1] = (input_image[:, :, 0] + input_image[:, :, 1] + input_image[:, :, 2]) / 3

    # luminosity method
    new_image[:, :, 2] = 0.21*input_image[:, :, 0] + 0.72*input_image[:, :, 1] + 0.07*input_image[:, :, 2]

    # built-in opencv function
    new_image[:, :, 3] = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)


    # ligtness method
    #new_image = (input_image.max(axis=2) + input_image.min(axis=2)) / 2

    # average method
    #new_image = (input_image[:, :, 0] + input_image[:, :, 1] + input_image[:, :, 2]) / 3

    # luminosity method
    #new_image = 0.21 * input_image[:, :, 0] + 0.72 * input_image[:, :, 1] + 0.07 * input_image[:, :, 2]

    # built-in opencv function
    #new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    new_image = np.zeros(input_image.shape)
    new_image[:, :, 0] = (input_image[:, :, 0] - input_image[:, :, 1]) / np.sqrt(2)
    new_image[:, :, 1] = (input_image[:, :, 0] + input_image[:, :, 1] - 2*input_image[:, :, 2]) / np.sqrt(6)
    new_image[:, :, 2] = (input_image[:, :, 0] + input_image[:, :, 1] + input_image[:, :, 2]) / np.sqrt(3)
    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    new_image = np.zeros(input_image.shape)
    norm = input_image[:, :, 0] + input_image[:, :, 1] + input_image[:, :, 2]
    new_image[:, :, 0] = input_image[:, :, 0] / norm
    new_image[:, :, 1] = input_image[:, :, 1] / norm
    new_image[:, :, 2] = input_image[:, :, 2] / norm
    return new_image
