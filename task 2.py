from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from skimage import io, filters, measure, morphology, img_as_float
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops, regionprops_table
from skimage.util import random_noise
from skimage.feature import match_template, peak_local_max


def graph(grayscale_image, grayscale_image_temp, highlight, local_max):
    '''defines function that plots the input image and template image
       plots a highlighter around the target'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    ax1.imshow(grayscale_image_temp, cmap=plt.cm.gray)
    ax2.imshow(grayscale_image, cmap=plt.cm.gray)
    ax3.imshow(grayscale_image, cmap=plt.cm.gray)
    
    ax3.plot(local_max[:, 1], local_max[:, 0], 'r.')
    
    ax1.set_title('template_tie')
    ax2.set_title('tie')
    ax3.set_title('local maxima')
        
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()

    ax3.autoscale(False)

    ax2.add_patch(highlight)
    
    plt.show()


def main():
    '''defines function which reads in the image template and input image
       converts images to grayscale
       tries to match the template image to the input image and returns a normalised value
       creates a variable for highlighting the target'''
    image_file = 'tie.jpg'
    image_template_file = 'template_tie.jpg'
    image = io.imread(image_file)
    image_template = io.imread(image_template_file)
    
    grayscale_image = rgb2gray(image)
    grayscale_image_temp = rgb2gray(image_template)
    image_float = img_as_float(grayscale_image)
    
    result = match_template(grayscale_image, grayscale_image_temp)
    pos = np.unravel_index(np.argmax(result), result.shape)
    x, y = pos[::-1]
    local_max = peak_local_max(image_float, min_distance=35)
    
    wtie, htie = grayscale_image_temp.shape
    highlight = plt.Rectangle((x, y), wtie, htie, edgecolor='r', facecolor='none')
    
    graph(grayscale_image, grayscale_image_temp, highlight, local_max)
    
    
if __name__ == "__main__":
    main()