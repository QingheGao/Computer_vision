import cv2
import numpy as np
from keypoint_matching import keypoint_matching
from RANSAC import RANSAC
import matplotlib.pyplot as plt


def stitch(img, img2):
    kp1, kp2, matches = keypoint_matching(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    aligned_img, _ = RANSAC(img, kp1, kp2, matches, iters=2000, P=3)

    stitched_img = np.zeros((max(img2.shape[0], aligned_img.shape[0]), max(img2.shape[1], aligned_img.shape[1]), 3))
    stitched_img[:aligned_img.shape[0], :aligned_img.shape[1]] = aligned_img
    stitched_img[:img2.shape[0], :img2.shape[1]] = img2
    stitched_img /= 255

    return stitched_img


if __name__ == "__main__":
    img = cv2.imread("./right.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread("./left.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    stitched_img = stitch(img, img2)

    plt.imshow(stitched_img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("stitched_img.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()