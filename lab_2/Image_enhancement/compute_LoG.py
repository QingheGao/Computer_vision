from gauss2D import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

def compute_LoG(image, LOG_type):

    if LOG_type == 1:
        #method 1
        gauss = gauss2D(0.5,5)
        smooth = cv2.filter2D(image,-1,gauss)
        imOut = cv2.Laplacian(smooth, -1, ksize=3)



    elif LOG_type == 2:
        #method 2
        print('Not implemented\n')
        gray_lap = cv2.Laplacian(image, -1, ksize=5,scale=0.5)
        imOut = cv2.convertScaleAbs(gray_lap)


    elif LOG_type == 3:
        #method 3
        gauss1 = gauss2D(0.6, 5)
        gauss2 = gauss2D(0.5, 5)

        smooth1 = cv2.filter2D(image, -1, gauss1)
        imOut1 = cv2.Laplacian(smooth1, -1, ksize=5)

        smooth2 = cv2.filter2D(image, -1, gauss2)
        imOut2 = cv2.Laplacian(smooth2, -1, ksize=5)
        imOut = imOut2 - imOut1



    return imOut

img_gray = cv2.imread('./images/image2.jpg')[...,0].astype(np.float32)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

plt.subplot(131)
imOut = compute_LoG(img_gray, LOG_type=1)
plt.imshow(imOut, cmap='gray')
plt.title('Gaussian + Laplacian')
plt.xticks([])
plt.yticks([])
plt.subplot(132)
imOut = compute_LoG(img_gray, LOG_type=2)
plt.imshow(imOut, cmap='gray')
plt.title('LoG')
plt.xticks([])
plt.yticks([])
plt.subplot(133)
imOut = compute_LoG(img_gray, LOG_type=3)
plt.imshow(imOut, cmap='gray')
plt.title('DoG')
plt.xticks([])
plt.yticks([])
#
# plt.savefig('./figure/threemethods.png',dpi=400,bbox_inches='tight')
plt.show()
