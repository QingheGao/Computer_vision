import os
import numpy as np
import glob
from matplotlib import cm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_syn_images(image_dir='./SphereGray5/', channel=0):
    ### the doing the color images, channel = 3
    files = os.listdir(image_dir)

    #files = [os.path.join(image_dir, f) for f in files]
    nfiles = len(files)
    
    image_stack = None
    V = 0
    Z = 0.5
    
    for i in range(nfiles):
        # read light direction from image name
        # read input image
        im = cv2.imread(os.path.join(image_dir, files[i]))
        im = im[:,:,channel]

        # stack at third dimension
        if image_stack is None:
            h, w = im.shape
            print('Image size (H*W): %d*%d' %(h,w) )
            image_stack = np.zeros([h, w, nfiles], dtype=int)
            V = np.zeros([nfiles, 3], dtype=np.float64)

        image_stack[:,:,i] = im

        # read light direction from image name
        X = np.double(files[i][(files[i].find('_')+1):files[i].rfind('_')])
        Y = np.double(files[i][files[i].rfind('_')+1:files[i].rfind('.png')])
        V[i, :] = [-X, Y, Z]

    ## for color one
    # for i in range(nfiles):
    #     # read light direction from image name
    #     # read input image
    #     im = cv2.imread(os.path.join(image_dir, files[i]))
    #
    #     # im = im[:, :, channel]
    #
    #     # stack at third dimension
    #     if image_stack is None:
    #         h, w, _ = im.shape
    #         print('Image size (H*W): %d*%d' % (h, w))
    #         image_stack = np.zeros([h, w, channel, nfiles], dtype=float)
    #         V = np.zeros([nfiles, 3], dtype=np.float64)
    #
    #     image_stack[:, :, :, i] = np.asarray(im, dtype=np.float) / 255.0
    #
    #     # read light direction from image name
    #     X = np.double(files[i][(files[i].find('_') + 1):files[i].rfind('_')])
    #     Y = np.double(files[i][files[i].rfind('_') + 1:files[i].rfind('.png')])
    #     V[i, :] = [-X, Y, Z]

        
    # normalization
    image_stack = np.double(image_stack)
    min_val = np.min(image_stack)
    max_val = np.max(image_stack)
    image_stack = (image_stack - min_val) / (max_val - min_val)
    normV = np.tile(np.sqrt(np.sum(V ** 2, axis=1, keepdims=True)), (1, V.shape[1]))
    scriptV = V / normV
    
    return image_stack, scriptV
    
    
def load_face_images(image_dir='./yaleB02/'):
    num_images = 57
    filename = os.path.join(image_dir, 'yaleB02_P00_Ambient.pgm')
    ambient_image = cv2.imread(filename, -1)
    h, w = ambient_image.shape

    # get list of all other image files
    import glob 
    d = glob.glob(os.path.join(image_dir, 'yaleB02_P00A*.pgm'))
    import random
    d = random.sample(d, num_images)
    filenames = [os.path.basename(x) for x in d]

    ang = np.zeros([2, num_images])
    image_stack = np.zeros([h, w, num_images])

    for j in range(num_images):
        ang[0,j], ang[1,j] = np.double(filenames[j][12:16]), np.double(filenames[j][17:20])
        image_stack[...,j] = cv2.imread(os.path.join(image_dir, filenames[j]), -1) - ambient_image


    x = np.cos(np.pi*ang[1,:]/180) * np.cos(np.pi*ang[0,:]/180)
    y = np.cos(np.pi*ang[1,:]/180) * np.sin(np.pi*ang[0,:]/180)
    z = np.sin(np.pi*ang[1,:]/180)
    scriptV = np.array([y,z,x]).transpose(1,0)

    image_stack = np.double(image_stack)
    image_stack[image_stack<0] = 0
    min_val = np.min(image_stack)
    max_val = np.max(image_stack)
    image_stack = (image_stack - min_val) / (max_val - min_val)
    
    return image_stack, scriptV
    
    
def show_results(albedo, normals, height_map, SE):
    # Stride in the plot, you may want to adjust it to different images
    stride = 1
    
    #showing albedo map
    fig = plt.figure()
    albedo_max = albedo.max()
    albedo_max = 1
    albedo = albedo / albedo_max
    print(albedo.shape)
    plt.title('SphereColor Albedo')
    plt.imshow(albedo, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./plot/SphereColor_Albedo.jpg',bbox_inches='tight',dpi=500)
    plt.show()

    #showing normals as three separate channels
    figure = plt.figure()
    ax1 = figure.add_subplot(131)
    ax1.imshow(normals[..., 0])
    plt.xticks([])
    plt.yticks([])
    plt.title('Normal X')
    ax2 = figure.add_subplot(132)
    ax2.imshow(normals[..., 1])
    plt.xticks([])
    plt.yticks([])
    plt.title('Normal Y')
    ax3 = figure.add_subplot(133)
    ax3.imshow(normals[..., 2])
    plt.title('Normal Z')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./plot/25imagesthreenormal.jpg', bbox_inches='tight', dpi=500)


## for color one
    # figure = plt.figure()
    # normals_nan = normals[~np.isnan(normals)]
    # normals_ = (normals - np.min(normals_nan )) / (np.max(normals_nan ) - np.min(normals_nan ))
    # normals_[np.isnan(normals)] = np.nan
    # normals_ = np.nan_to_num(normals_)
    # for i in range(albedo.shape[2]):
    #     ax = figure.add_subplot(141)
    #     ax.imshow(albedo[:,:,i],cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title('Albedo channel'+str(i+1))
    #     ax1 = figure.add_subplot(142)
    #     ax1.imshow(normals_[:,:,i, 0],cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title('Normal X')
    #     ax2 = figure.add_subplot(143)
    #     ax2.imshow(normals_[:,:,i, 1],cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title('Normal Y')
    #     ax3 = figure.add_subplot(144)
    #     ax3.imshow(normals_[:,:,i, 2],cmap='gray')
    #     plt.title('Normal Z')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.savefig('./plot/Monkey_color_channel'+str(i)+'.jpg', bbox_inches='tight', dpi=500)
    #     plt.show()


    #meshgrid

    '''
    =============
    You could further inspect the shape of the objects and normal directions by using plt.quiver() function.
    =============
    '''


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, _ = np.meshgrid(np.arange(0, np.shape(normals)[0], stride),
                          np.arange(0, np.shape(normals)[1], stride),
                          np.arange(0, np.shape(normals)[2], stride))
    X = X[..., 0]
    Y = Y[..., 0]

    x = X[::15,::15]
    y = Y[::15,::15]

    h, w = height_map[::20, ::20].shape
    x, y = np.meshgrid(np.arange(w),
                       np.arange(h))
    z = np.nan_to_num(height_map[::20, ::20])
    u = np.nan_to_num(normals[::20, ::20, 0])
    v = np.nan_to_num(normals[::20, ::20, 1])
    w = np.nan_to_num(normals[::20, ::20, 2])

    plt.title('25 images')
    ax.quiver(x, y, z, -u, -v, w, length=5, arrow_length_ratio=.4,color = 'black')
    ax.set_zlim(-10,50)
    ax.view_init(20, 50)

    # plt.savefig('./plot/25surfacenormarrow.jpg', bbox_inches='tight', dpi=500)
    plt.show()

    # #plotting the SE
    H = SE[::stride,::stride]
    fig = plt.figure()
    ax = fig.add_subplot(221, projection = '3d')
    ax.plot_surface(X,Y, H.T,color='red')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(30, 10)
    plt.title('25 image SE ')
    # plt.savefig('./plot/5SE_average.jpg', bbox_inches='tight', dpi=500)
    ax1 = fig.add_subplot(222)
    ax1.imshow(normals[..., 0],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Normal X')
    ax2 = fig.add_subplot(223)
    ax2.imshow(normals[..., 1],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Normal Y')
    ax3 = fig.add_subplot(224)
    ax3.imshow(normals[..., 2],cmap='gray')
    plt.title('Normal Z')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./plot/25SEE.jpg', bbox_inches='tight', dpi=500)
    plt.show()



    # plotting model geometry
    H = height_map[::stride,::stride]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x_scale = 2
    y_scale = 2
    z_scale = 2.5

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    ax.plot_surface(X,Y, H.T,cmap='PuBu_r')
    ax.view_init(30, 50)
    plt.title('Height map average')
    plt.savefig('./plot/25height_average.jpg', bbox_inches='tight', dpi=500)

    plt.show()
    #
    ##### for color one


    #
    # X, Y, _ = np.meshgrid(np.arange(0, np.shape(normals)[0], stride),
    #                       np.arange(0, np.shape(normals)[1], stride),
    #                       np.arange(0, np.shape(normals)[2], stride))
    # X = X[..., 0]
    # Y = Y[..., 0]
    # fig = plt.figure()
    # ax = fig.add_subplot(131, projection='3d')
    # ax.plot_surface(X, Y, height_map[::stride, ::stride, 0].T)
    # ax.view_init(30, 50)
    # plt.title('Height map Channel 1')
    # ax = fig.add_subplot(132, projection='3d')
    # ax.plot_surface(X, Y, height_map[::stride, ::stride, 1].T)
    # ax.view_init(30, 50)
    # plt.title('Height map Channel 2')
    # ax = fig.add_subplot(133, projection='3d')
    # ax.plot_surface(X, Y, height_map[::stride, ::stride, 1].T)
    # ax.view_init(30, 50)
    # plt.title('Height map Channel 3')
    # plt.show()
    #
    #

