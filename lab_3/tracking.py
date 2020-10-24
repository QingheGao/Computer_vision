from lucas_kanade import *
from harris_corner_detection import *
import matplotlib.animation as animation
import os
from os import listdir
from os.path import join
import imageio

def load_images(path):
    images = []
    files = sorted(listdir(path))

    for f in files:
        img = cv2.imread(join(path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
        images.append(img)

    return images



def tracking(dir = "./pingpong"):
    images = load_images(dir)
    H, r, c = harris_detector(images[0], 130000, 9)

    for i in range(len(images)-1):
        v = lucas_kanade(images[i], images[i+1], patch_size=25, features=(r, c))
        plt.figure(figsize=(8, 8))
        plt.imshow(images[i], interpolation='none', animated=True, cmap="gray")

        plt.quiver(c, r, v[:, 0, 0], v[:, 1, 0], angles="xy", scale=3,
                   color="r")
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('./ping_tracking/'+str(i)+'.png', bbox_inches='tight')
        plt.show()

        #### for toy_tracking
        # r = (r + np.floor(2 * v[:, 1, 0].round(1))).astype(np.int32)
        # c = (c + np.floor(2 * v[:, 0, 0].round(1))).astype(np.int32)

        ### for pingpong
        r = np.ceil((r + 3*v[:, 1, 0].round(1))).astype(np.int32)
        c = np.ceil((c + 3*v[:, 0, 0].round(1))).astype(np.int32)




#### create gif
def create_gif(image_dir='./ping_tracking'):
    images = []
    files = os.listdir(image_dir)
    files.sort(key = lambda x:int(x[:-4]))
    nfiles = len(files)
    for i in range(nfiles):
        images.append(imageio.imread(os.path.join(image_dir, files[i])))
    imageio.mimsave('./gif/pingpongtest.gif', images)


if __name__ == "__main__":
    tracking("./pingpong")
    create_gif('./ping_tracking')

