from gauss1D import *

def gauss2D( sigma , kernel_size ):
    ## solution
    gx = gauss1D(sigma , kernel_size )
    gy = gauss1D(sigma , kernel_size )
    G = gx.T *gy
    print(G)
    return G


if __name__ == '__main__':
    gauss2D(2 , 5)