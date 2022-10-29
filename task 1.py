import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data, io, filters, feature
from skimage.color import rgb2gray
from skimage.util import random_noise


def graph(grayscale_resized, noisy_image, laplace_filter, canny_filter):
    '''Defines a function to plots the grayscale image and noisy image with and without laplace and canny filters applied'''
    fig, ax = plt.subplots(1, 4, figsize=(8, 4))

    ax[0].imshow(grayscale_resized)
    ax[1].imshow(noisy_image)
    ax[2].imshow(laplace_filter)
    ax[3].imshow(canny_filter)
    
    ax[0].set_title('Grayscale Resized')
    ax[1].set_title('Noisy image')
    ax[2].set_title('laplace filter')
    ax[3].set_title('Canny filter')

    fig.tight_layout()
    
    plt.show()

    
def main():
    ''''Defines the main functioin which reads the image and converts to grayscale, resizes and applies laplace and canny filters
        Prints the min, max and mean values of the grayscale image'''
    filename = 'facepic.jpg'
    original = io.imread(filename)
    grayscale = rgb2gray(original)
    grayscale_resized = resize(grayscale, (512, 512))
    noisy_image = random_noise(grayscale_resized, mode='gaussian')
    laplace_filter = filters.laplace(noisy_image)
    canny_filter = feature.canny(noisy_image, mode='reflect')
    print(f"Min: {grayscale_resized.min()} \nMax: {grayscale_resized.max()}\nMean: {grayscale_resized.mean()}")
    graph(grayscale_resized, noisy_image, laplace_filter, canny_filter)
    

if __name__ == "__main__":
    main()