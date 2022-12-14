import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.feature import match_template, peak_local_max


def graph(grayscale_image, grayscale_image_temp, highlight, local_max):
    '''defines function that plots a graph
       shows the grayscale template image and grayscale input image
       plots a highlighter around any instance of the template image inside the input image
       plots peak local maxima for all instances of the template image'''
    #create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    #show images
    ax1.imshow(grayscale_image_temp, cmap=plt.cm.gray)
    ax2.imshow(grayscale_image, cmap=plt.cm.gray)
    ax3.imshow(grayscale_image, cmap=plt.cm.gray)
    
    #plot peak local maximas
    ax3.plot(local_max[:, 1], local_max[:, 0], 'r.')
    
    #set titles to all images
    ax1.set_title('template_tie')
    ax2.set_title('tie')
    ax3.set_title('local maxima')
    
    #set all axis off
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()

    #highlight the detected match
    ax2.add_patch(highlight)
    
    plt.show()


def main():
    '''defines function which reads in the image template and input image
       converts images to grayscale
       matches the template image to the input image using template matching function
       creates a highlight around any instance of the template image
       calculates the peak local maximas to find all instances of the template image
       calls graph function'''
    #read in images
    image_file = 'tie.jpg'
    image_template_file = 'template_tie.jpg'
    image = io.imread(image_file)
    image_template = io.imread(image_template_file)
    
    #convert images to grayscale
    grayscale_image = rgb2gray(image)
    grayscale_image_temp = rgb2gray(image_template)
    image_float = img_as_float(grayscale_image)
    
    #match template image and calculate peak local maxima
    result = np.array(match_template(grayscale_image, grayscale_image_temp))
    pos = np.unravel_index(np.argmax(result), result.shape)
    x, y = pos[::-1]
    local_max = peak_local_max(image_float, min_distance=1, threshold_rel=0.5)
    
    #calculate width and height of the template and create a highlight around it
    wtie, htie = grayscale_image_temp.shape
    highlight = plt.Rectangle((x, y), wtie, htie, edgecolor='r', facecolor='none')
    
    #call graph function to display graph
    graph(grayscale_image, grayscale_image_temp, highlight, local_max)
    
    
#run code
if __name__ == "__main__":
    main()