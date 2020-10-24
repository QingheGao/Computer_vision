import numpy as np

def estimate_alb_nrm( image_stack, scriptV, shadow_trick= False):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    # h, w, channel, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])

    # albedo = np.zeros([h, w,channel])
    # normal_surface = np.zeros([h, w, channel, 3])
    # normal = np.zeros([h, w, 3])
    
    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
        
    """
    for x in range(h):
        for y in range(w):
            i = np.array(image_stack[x, y]).reshape((-1, 1))
            scriptI = np.diag(i.T[0])
            if shadow_trick == True:
                g = np.linalg.lstsq(np.dot(scriptI, scriptV), np.dot(scriptI, i))[0]
            else:
                g = np.linalg.lstsq(scriptV, i)[0]
            albedo[x, y] = np.linalg.norm(g)
            normal[x, y] = (g / np.linalg.norm(g)).T

    # for x in range(h):
    #     for y in range(w):
    #         for c in range(channel):
    #             i = np.array(image_stack[x, y, c]).reshape((-1, 1))
    #             scriptI = np.diag(i.T[0])
    #             if shadow_trick == True:
    #                 g = np.linalg.lstsq(np.dot(scriptI, scriptV), np.dot(scriptI, i))[0]
    #             else:
    #                 g = np.linalg.lstsq(scriptV, i)[0]
    #             albedo[x, y ,c] = np.linalg.norm(g)
    #             normal_surface[x,y,c] = (g / np.linalg.norm(g)).T
    #
    # normal[:,:,0] = (np.mean(normal_surface[:, :, :,0],axis=2))
    # normal[:, :, 1] = (np.mean(normal_surface[:, :, :,1],axis=2))
    # normal[:, :, 2] = (np.mean(normal_surface[:, :, :,2],axis=2))
    print(albedo.shape,normal.shape)



    
    return albedo, normal
    
if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick= False)