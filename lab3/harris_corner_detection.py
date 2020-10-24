import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import imutils

def harris_detector(image, threshold, n):
    """
    :param image: input image
    :param threshold: threshold for corner value
    :param n: window size to search for local maxima
    :return:
    """
    Gx = np.array([[-1, 0, 1]])
    Gy = Gx.T

    Ix = convolve2d(image, Gx, boundary="symm", mode="same")
    Iy = convolve2d(image, Gy, boundary="symm", mode="same")

    A = gaussian_filter(Ix ** 2, 1.5,5)
    B = gaussian_filter(Ix * Iy, 1.5,5)
    C = gaussian_filter(Iy ** 2, 1.5,5)

    H = (A * C - B ** 2) - 0.04 * (A + C) ** 2

    r = []
    c = []
    for i in range(n // 2, H.shape[0] - n // 2):
        for j in range(n // 2, H.shape[1] - n // 2):
            if H[i, j] == H[i - n // 2:i + n // 2 + 1, j - n // 2:j + n // 2 + 1].max() and H[i, j] > threshold:
                r.append(i)
                c.append(j)

    return H, np.array(r), np.array(c)

if __name__ == "__main__":
    img = cv2.imread("./pingpong/0000.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)

    img2 = cv2.imread("./person_toy/00000001.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img2 = img2.astype(np.float32)

    _,r,c = harris_detector(img,130000,9)
    plt.imshow(img,cmap='gray')
    plt.scatter(c,r,color ='yellow',marker='+')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
    _, r, c = harris_detector(img2, 50000, 9)
    plt.imshow(img2, cmap='gray')
    plt.scatter(c, r, color='yellow', marker='+')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

    ##90
    img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    _, r, c = harris_detector(img_rotate_90_clockwise, 130000, 9)
    plt.imshow(img_rotate_90_clockwise,cmap='gray')
    plt.scatter(c, r, marker='+', color='yellow', linewidths=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    ##45

    r45 = imutils.rotate(img,angle=45)
    _,r,c = harris_detector(r45, 130000, 9)
    plt.imshow(r45,cmap='gray')
    plt.scatter(c,r,marker='+',color= 'yellow',linewidths=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()



