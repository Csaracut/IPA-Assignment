import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.feature import peak_local_max


def graph(grayscale_image):
    fig, (ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    ax1.imshow(grayscale_image, cmap=plt.cm.gray)

    ax1.set_title('raisins')

    ax1.set_axis_off()

    plt.show()




def main():
    image_file = 'raisins'
    image = io.imread(image_file)
    
    grayscale_image = rgb2gray(image)
    
    graph(grayscale_image)

if __name__ == "__main__":
    main()