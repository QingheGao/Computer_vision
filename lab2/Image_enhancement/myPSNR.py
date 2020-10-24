import matplotlib.pyplot as plt
import numpy as np
def myPSNR( orig_image, approx_image ):
    # print('Not implemented\n')

    h, w = orig_image.shape

    MSE = np.sum((orig_image - approx_image)**2)/(h*w)

    PSNR = 20 * np.log10(max(map(max,orig_image))/np.sqrt(MSE))


    return PSNR


# orig_image = plt.imread('./images/image1_saltpepper.jpg').astype(np.float32)
# approx_image = plt.imread('./images/image1.jpg').astype(np.float32)
# print(myPSNR( orig_image, approx_image ))
