import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def graph(overlay, object):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.imshow(overlay)
    
    ax1.set_title('image_label_overlay')
    
    ax1.set_axis_off()

    for region in regionprops(object):
        if region.area >= 50:
            min_y, min_x, max_y, max_x = region.bbox
            
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red')
            
            ax1.add_patch(rect)

    plt.tight_layout()
    plt.show()


def main():
    image_file = 'raisins.jpg'
    image = io.imread(image_file)
    
    grayscale_image = rgb2gray(image)
    
    threshold = threshold_otsu(grayscale_image)
    border_objects = closing(grayscale_image < threshold)
    
    clear_objects= clear_border(border_objects)
    
    object = label(clear_objects)
    
    overlay = label2rgb(object, image=grayscale_image, bg_label=0)
    
    graph(overlay, object)


if __name__ == "__main__":
    main()