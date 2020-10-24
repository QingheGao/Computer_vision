import numpy as np

def gauss1D(sigma , kernel_size ):

    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    value = np.floor(kernel_size/2)


    for index, x in enumerate(np.arange(-value,value+1,1)):
        G[0][index]= (1/(sigma * np.sqrt(2*np.pi)))*np.exp(-x**2/(2*sigma**2))
    G = G/np.sum(G)
    return G

if __name__ == '__main__':
    gauss1D(2 , 5)