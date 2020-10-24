import numpy as np
import matplotlib.pyplot as plt
import cv2
from keypoint_matching import keypoint_matching


def transform_corners(transform, img):
    """
    Returns the coordinates of the transformed corners of img
    """
    corner_A = np.zeros((8, 6))
    corner_A[np.arange(0, 8, 2), 4] = 1
    corner_A[np.arange(1, 8, 2), 5] = 1

    # top left corner
    corner_A[[0, 1], [0, 2]] = 0
    corner_A[[0, 1], [1, 3]] = 0

    # bottom left corner
    corner_A[[2, 3], [0, 2]] = 0
    corner_A[[2, 3], [1, 3]] = img.shape[0] - 1

    # top right corner
    corner_A[[4, 5], [0, 2]] = img.shape[1] - 1
    corner_A[[4, 5], [1, 3]] = 0

    # bottom right corner
    corner_A[[6, 7], [0, 2]] = img.shape[1] - 1
    corner_A[[6, 7], [1, 3]] = img.shape[0] - 1

    transformed_corners = corner_A.dot(transform)

    return [(transformed_corners[0].item(), transformed_corners[1].item()),
            (transformed_corners[2].item(), transformed_corners[3].item()),
            (transformed_corners[4].item(), transformed_corners[5].item()),
            (transformed_corners[6].item(), transformed_corners[7].item())]


def construct_matrices(kp1, kp2, matches):
    """
    Constructs the matrices A and b, that are necessary to solve the linear system, from the matched keypoints
    """
    N = len(matches)

    A = np.zeros((2 * N, 6))
    b = np.zeros((2 * N, 1))

    for j in range(N):
        A[2 * j] = [kp1[matches[j].queryIdx].pt[0], kp1[matches[j].queryIdx].pt[1], 0, 0, 1, 0]
        A[2 * j + 1] = [0, 0, kp1[matches[j].queryIdx].pt[0], kp1[matches[j].queryIdx].pt[1], 0, 1]
        b[2 * j] = kp2[matches[j].trainIdx].pt[0]
        b[2 * j + 1] = kp2[matches[j].trainIdx].pt[1]

    return A, b


def transform_image(transform, img, method="opencv"):
    if method == "opencv":
        # transform whole image
        corner_coords = transform_corners(transform, img)
        transform = np.array([[transform[0].item(), transform[1].item(), transform[4].item()],
                              [transform[2].item(), transform[3].item(), transform[5].item()]])

        warp_dst = cv2.warpAffine(img, transform,
                                  (int(max(corner_coords[0][0], corner_coords[1][0], corner_coords[2][0], corner_coords[3][0])),
                                   int(max(corner_coords[0][1], corner_coords[1][1], corner_coords[2][1], corner_coords[3][1]))))

        return warp_dst
    else:
        # fill matrix A with image coordinates
        A = np.zeros((2*img.size, 6))
        counter = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                A[2*counter] = [j, i, 0, 0, 1, 0]
                A[2*counter + 1] = [0, 0, j, i, 0, 1]

                counter += 1

        # transform coords
        transformed_coords = A.dot(transform)

        # apply nearest neighbour interpolation
        nn_coords = np.around(transformed_coords).astype(np.int32)

        # shift the coordinates in case we get negative ones, so we can properly insert the old image into new matrix
        x_offset = abs(min(0, nn_coords[np.arange(0, nn_coords.size, 2)].min()))
        y_offset = abs(min(0, nn_coords[np.arange(1, nn_coords.size, 2)].min()))
        nn_coords[np.arange(0, nn_coords.size, 2)] += x_offset
        nn_coords[np.arange(1, nn_coords.size, 2)] += y_offset

        # create a blank matrix + insert the old image at the transformed coordinates
        new_img = np.zeros((nn_coords[np.arange(1, nn_coords.size, 2)].max() + 1,
                            nn_coords[np.arange(0, nn_coords.size, 2)].max() + 1))
        new_img[nn_coords[np.arange(1, nn_coords.size, 2)], nn_coords[np.arange(0, nn_coords.size, 2)]] = img.reshape(-1, 1)

        new_img = new_img[y_offset:, x_offset:]

        # for x in np.arange(1,new_img.shape[0]-1):
        #     for y in np.arange(1, new_img.shape[0] - 1):
        #         if new_img[x,y] == 0:
        #             new_img[x,y] = (new_img[x+1,y]+new_img[x-1,y]+new_img[x,y-1]+new_img[x,y+1])/4

        return new_img


def RANSAC(img1, kp1, kp2, matches, iters=10, P=10, method="opencv", display=False):
    best_n_inliners = 0
    best_iteration = 0
    best_transform = None
    A, b = construct_matrices(kp1, kp2, matches)

    indices = 2*np.arange(0, len(matches))

    for i in range(iters):
        # pick P random matches
        np.random.shuffle(indices)
        idx = indices[:P]
        idx = np.hstack((idx, idx + 1))

        # construct the matrices needed to solve the linear system
        sub_A = A[idx]
        sub_b = b[idx]

        # solve the linear system
        x = np.linalg.pinv(sub_A) @ sub_b

        # transform all keypoints
        transformed_b = A.dot(x)

        # count # of inliners
        distance = np.abs(b - transformed_b)
        distance = np.sqrt(distance[np.arange(0, len(distance), 2)]**2 + distance[np.arange(1, len(distance), 2)]**2)
        n_inliners = np.sum(distance < 10)

        # display 10 random matches after each iteration (Note: this produces A LOT of sequential plot)
        if display:
            print("Number of inliners:", n_inliners)

            # create new kp2 and create new matches + filter kp1 to only include matched keypoints
            new_kp1 = []
            new_kp2 = []
            new_matches = []

            for j in range(b.shape[0] // 2):
                new_kp1.append(cv2.KeyPoint(A[2 * j, 0], A[2 * j, 1], 1))
                new_kp2.append(cv2.KeyPoint(transformed_b[2 * j], transformed_b[2 * j + 1], 1))
                new_matches.append(cv2.DMatch(j, j, j))

            np.random.shuffle(new_matches)
            img3 = cv2.drawMatches(img1, new_kp1, img2, new_kp2, new_matches[:10], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3)
            plt.show()

        if n_inliners > best_n_inliners:
            # display images with transformed keypoints on second image
            best_n_inliners = n_inliners
            best_transform = x
            best_iteration = i

    # transform whole image
    warp_dst = transform_image(best_transform, img1, method=method)
    return warp_dst, best_iteration


def find_avg_best_iterations(img, img2, N, P):
    kp1, kp2, matches = keypoint_matching(img, img2, display=False)
    results = []

    for i in range(N):
        print(i)
        _, best_iteration = RANSAC(img, kp1, kp2, matches, 1000, P=P)
        results.append(best_iteration)

    print("Average # of iterations to get best results:", np.mean(results))
    print("Standard deviation:", np.std(results))


if __name__ == "__main__":
    img = cv2.imread("./boat2.pgm")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("./boat1.pgm")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #find_avg_best_iterations(img, img2, 1000, 3)

    kp1, kp2, matches = keypoint_matching(img, img2, display=False)
    aligned_img, _ = RANSAC(img, kp1, kp2, matches, iters=1000, P=3, display=False, method="opencv")
    matched_img = cv2.drawMatches(img2, [], aligned_img.astype(np.uint8), [], [], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Transformed boat2.pgm")
    plt.show()

    kp1, kp2, matches = keypoint_matching(img2, img, display=False)
    aligned_img, _ = RANSAC(img2, kp1, kp2, matches, iters=1000, P=3, display=False, method="opencv")
    matched_img = cv2.drawMatches(img, [], aligned_img.astype(np.uint8), [], [], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Transformed boat1.pgm")
    plt.show()
