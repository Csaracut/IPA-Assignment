import matplotlib.pyplot as plt
import matplotlib.patches as matpatch
from skimage.transform import resize
from skimage import data, io, filters, measure, morphology
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops, regionprops_table
from skimage.util import random_noise


def graph():
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))

    
 
    plt.show()

def main():
    image = 'tie.jpg'
    image_template = 'template_tie.jpg'
    
    grayscale_image = rgb2gray(image)
    grayscale_image_temp = rgb2gray(image_template)
    
    
    
    
    
    
if __name__ == "__main__":
    main()