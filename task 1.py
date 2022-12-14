import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io, filters, feature
from skimage.color import rgb2gray
from skimage.util import random_noise


def graph(grayscale_resized, noisy_image, laplace_filter, canny_filter):
    '''Defines a function to plots the grayscale image and noisy image with and without laplace and canny filters applied'''
    #create plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 4))

    #show all images
    ax1.imshow(grayscale_resized, cmap=plt.cm.gray)
    ax2.imshow(noisy_image, cmap=plt.cm.gray)
    ax3.imshow(laplace_filter, cmap=plt.cm.gray)
    ax4.imshow(canny_filter, cmap=plt.cm.gray)
    
    #set titles to all images
    ax1.set_title('Grayscale Resized')
    ax2.set_title('Noisy image')
    ax3.set_title('laplace filter')
    ax4.set_title('Canny filter')

    fig.tight_layout()
    
    plt.show()

    
def main():
    ''''Defines the main functioin which reads the image and 
        converts image to grayscale, resizes and applies laplace and canny filters
        Prints the min, max and mean values of the grayscale image
        calls graph function'''
    #read in image
    image_file = 'facepic.jpg'
    original = io.imread(image_file)
    
    #convert to grayscale
    grayscale = rgb2gray(original)
    grayscale_resized = resize(grayscale, (512, 512))
    
    #apply noise and filters
    noisy_image = random_noise(grayscale_resized, mode='gaussian')
    laplace_filter = filters.laplace(noisy_image)
    canny_filter = feature.canny(noisy_image, mode='reflect')
    
    #print out min, max and mean grayscale values
    print(f"Min: {grayscale_resized.min()} \nMax: {grayscale_resized.max()}\nMean: {grayscale_resized.mean()}")
    
    #call graph function to display graph
    graph(grayscale_resized, noisy_image, laplace_filter, canny_filter)
    
    
#run code
if __name__ == "__main__":
    main()