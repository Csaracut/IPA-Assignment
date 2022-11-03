import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing


def graph(overlay, label_object, min_area):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.imshow(overlay)
    
    ax1.set_title('image_label_overlay')
    
    ax1.set_axis_off()

    for region in regionprops(label_object):
        if region.area >= 50 and region.area != min_area:
            min_y, min_x, max_y, max_x = region.bbox
            
            highlight = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red')
            
            ax1.add_patch(highlight)
        elif region.area == min_area:
            min_y, min_x, max_y, max_x = region.bbox
            
            highlight_smallest_raisin = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='blue')
            
            ax1.add_patch(highlight_smallest_raisin)
            
    plt.tight_layout()
    plt.show()
    

def main():
    image_file = 'raisins.jpg'
    image = io.imread(image_file)
    
    grayscale_image = rgb2gray(image)
    
    threshold = threshold_otsu(grayscale_image)
    border_objects = closing(grayscale_image < threshold)
    
    clear_objects= clear_border(border_objects)
    
    label_object = label(clear_objects)
    
    overlay = label2rgb(label_object, image=grayscale_image, alpha=0)
    
    areas = np.array([])
    count = 0
    centroids = np.array([])
    for region in regionprops(label_object):
        if region.area >= 50:
            centroids = np.append(centroids, region.centroid)
            areas = np.append(areas, region.area)
            count += 1
    
    min_area = np.sort(areas)[0]
    min_centroid = np.sort(centroids)[0]

    print(f"Number of raisins: {count}")
    print(f"Area of smallest raisin: {min_area}\nCentroid of smallest raisin: {min_centroid}")
     
    graph(overlay, label_object, min_area)


if __name__ == "__main__":
    main()