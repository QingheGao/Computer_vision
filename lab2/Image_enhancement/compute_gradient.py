import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
def compute_gradient(image):
    h,w = image.shape
    x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]])
    y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    Gx = scipy.signal.convolve2d(image,x)
    Gy = scipy.signal.convolve2d(image,y)
    # Gx = np.zeros((h - 3, w - 3))
    # Gy = np.zeros((h - 3, w - 3))
    # for i in range(h-3):
    #     for j in range(w-3):
    #         Gx[i, j] = np.sum(x * image[i:i + 3 ,j:j + 3])
    #         Gy[i, j] = np.sum(y * image[i:i + 3, j:j + 3])

    im_magnitude = np.sqrt(Gx**2 + Gy** 2)
    im_direction = np.arctan(Gy / Gx)

    plt.subplot(221)
    plt.imshow(Gx,cmap='gray')
    plt.title('Gradient in X direction ')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(222)
    plt.imshow(Gy, cmap='gray')
    plt.title('Gradient in Y direction ')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(223)
    plt.imshow(im_magnitude, cmap='gray')
    plt.title('Gradient magnitude ')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(224)
    plt.imshow(im_direction, cmap='gray')
    plt.title('Gradient direction ')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./figure/gradient', dpi=400, bbox_inches='tight')
    plt.show()

    return Gx, Gy, im_magnitude,im_direction

# im = plt.imread('./images/image2.jpg').astype(np.float32)

im = plt.imread('./images/image2.jpg').astype(np.float32)

compute_gradient(im)