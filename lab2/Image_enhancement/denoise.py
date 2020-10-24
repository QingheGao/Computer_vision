import cv2
import matplotlib.pyplot as plt
from myPSNR import *
import numpy as np
def denoise( image, kernel_type,**kwargs):
    if kernel_type == 'box':
        imOu = cv2.blur(image,ksize = kwargs['k'])


    elif kernel_type == 'median':
        imOu = cv2.medianBlur(image, ksize = kwargs['k'])

    elif kernel_type == 'gaussian':
        imOu = cv2.GaussianBlur(image,ksize=kwargs['k'],sigmaX=kwargs['sigma'])
    else:
        print('Operatio Not implemented')
    return imOu

original = cv2.imread('./images/image1.jpg')[...,0].astype(np.float32)
image_1 = cv2.imread('./images/image1_gaussian.jpg')[...,0]
# plt.imshow(image_1,cmap='gray')
# plt.show()

# im = denoise( image_1, kernel_type='gaussian', k =(3,3), sigma = 0.5)
# cv2.imshow('im',im )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

klist = [(3,3),(5,5),(7,7)]
kk = [3,5,7]
titlelist = ['3X3','5X5','7X7']
boxpsnr = []
medianpsnr = []
for index, i in enumerate(klist):
    im = denoise(image_1, kernel_type = 'box', k = i).astype(np.float32)
    boxpsnr.append(myPSNR(original, im))
    plt.imshow(im,cmap='gray')
    # plt.title('Box '+ titlelist[index])
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./figure/box'+titlelist[index]+'gau.png', dpi=400, bbox_inches='tight')
    plt.show()
    # im = denoise(image_1, kernel_type='median', k=kk[index]).astype(np.float32)
    # medianpsnr.append(myPSNR(original, im))
    # plt.imshow(im, cmap='gray')
    # # plt.title('Median ' + titlelist[index])
    # plt.xticks([])
    # plt.yticks([])
    # # plt.savefig('./figure/median'+titlelist[index]+'gau.png',dpi=400,bbox_inches='tight')
    # plt.show()
print(boxpsnr)

# sigmalist = [0.5]
# kk = [(3,3),(5,5),(7,7)]
# gaussianlist = []
# titlelist = ['3X3','5X5','7X7']
# z = 1
# for index,h in enumerate(kk):
#     im = denoise( image_1, kernel_type = 'gaussian', k = h ,sigma=1).astype(np.float32)
#     gaussianlist.append(myPSNR(original, im))
#     plt.imshow(im,cmap='gray')
#     plt.title('$\sigma$ = 0.5, ' + 'k = '+ titlelist[index])
#     plt.xticks([])
#     plt.yticks([])
#     z+=1
#     # plt.savefig('./figure/gaussian_sigma=0.5'+str(h)+'.png',dpi=400,bbox_inches='tight')
#     plt.show()
# #
# print(gaussianlist)

# cv2.medianBlur(image_1, ksize = 7)