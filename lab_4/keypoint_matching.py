import cv2
import matplotlib.pyplot as plt
import numpy as np


def keypoint_matching(img1, img2, n_features=None, display=False):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # if display is True we are displaying the ten best matches
    if display:
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:10], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(10, 5))
        plt.title("10 best matches")
        plt.imshow(matched_img, cmap="gray")
        #plt.savefig("ten_best_matches.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()

        np.random.shuffle(matches)
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(10, 5))
        plt.title("10 random matches")
        plt.imshow(matched_img, cmap="gray")
        #plt.savefig("ten_random_matches.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()

    # return <n_features> best matches if this number is given
    if n_features:
        matches = sorted_matches[:n_features]

    return kp1, kp2, matches


if __name__ == "__main__":
    img = cv2.imread("./boat1.pgm")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("./boat2.pgm")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    keypoint_matching(img, img2, display=True)