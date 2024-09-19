import numpy as np
import matplotlib.pyplot as plt


def show_image(image):
    plt.figure()
    plt.imshow(np.asarray(image))
    plt.show()
