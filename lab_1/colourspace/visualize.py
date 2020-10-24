def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

    
    #Opponent
    axs[0, 0].imshow(input_image / input_image.max())
    axs[0, 0].set_title('Opponent')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(input_image[:, :, 0], cmap="gray")
    axs[0, 1].set_title(r'$O_1$')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(input_image[:, :, 1], cmap="gray")
    axs[1, 0].set_title(r'$O_2$')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(input_image[:, :, 2], cmap="gray")
    axs[1, 1].set_title(r'$O_3$')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    """
    #Normalized RGB
    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('Normalized RGB')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(input_image[:, :, 0], cmap="gray")
    axs[0, 1].set_title(r'$r$')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(input_image[:, :, 1], cmap="gray")
    axs[1, 0].set_title(r'$g$')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(input_image[:, :, 2], cmap="gray")
    axs[1, 1].set_title(r'$b$')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    """

    """
    # HSV
    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('HSV')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(input_image[:, :, 0], cmap="gray")
    axs[0, 1].set_title(r'$H$')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(input_image[:, :, 1], cmap="gray")
    axs[1, 0].set_title(r'$S$')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(input_image[:, :, 2], cmap="gray")
    axs[1, 1].set_title(r'$V$')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    """

    """
    # YCbCr
    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('YCbCr')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(input_image[:, :, 0], cmap="gray")
    axs[0, 1].set_title(r'$Y$')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(input_image[:, :, 1], cmap="gray")
    axs[1, 0].set_title(r'$C_b$')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(input_image[:, :, 2], cmap="gray")
    axs[1, 1].set_title(r'$C_r$')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    """

    """
    #Gray
    axs[0, 0].imshow(input_image[:, :, 0], cmap="gray")
    axs[0, 0].set_title('Lightness')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(input_image[:, :, 1], cmap="gray")
    axs[0, 1].set_title(r'Average')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 0].imshow(input_image[:, :, 2], cmap="gray")
    axs[1, 0].set_title(r'Luminosity')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(input_image[:, :, 3], cmap="gray")
    axs[1, 1].set_title(r'OpenCV')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    """


    plt.show()
