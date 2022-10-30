import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as matpatch
from skimage.transform import resize, rotate
from skimage import data, io, filters, measure, morphology
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops, regionprops_table
from skimage.util import random_noise
from skimage.feature import match_template


def graph(grayscale_image, grayscale_image_temp, highlight):
    '''defines function that plots the input image and template image
       plots a highlighter around the target'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(grayscale_image_temp, cmap=plt.cm.gray)
    ax2.imshow(grayscale_image, cmap=plt.cm.gray)
 
    ax1.set_title('template_tie')
    ax2.set_title('tie')
    
    ax1.set_axis_off()
    ax2.set_axis_off()

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
    
    result = match_template(grayscale_image, grayscale_image_temp)
    pos = np.unravel_index(np.argmax(result), result.shape)
    x, y = pos[::-1]

    wtie, htie = grayscale_image_temp.shape
    highlight = plt.Rectangle((x, y), wtie, htie, edgecolor='r', facecolor='none')
    
    graph(grayscale_image, grayscale_image_temp, highlight)
    
    
if __name__ == "__main__":
    main()