import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing


def graph(overlay, label_object, min_area):
    '''defines function to create and display a graph
       highlights all occurences of raisins
       highlights the smallest raisin in size'''
    #create graph
    fig, ax1 = plt.subplots(figsize=(10, 6))

    #show image
    ax1.imshow(overlay)
    
    #set title to image
    ax1.set_title('image_label_overlay')
    
    #set axis off
    ax1.set_axis_off()

    #iterate through each region and highlight each valid region (i.e. ignore regions with areas smaller than 50) and give a unique highlight to the smallest region
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
    '''defines main function that reads in image
       converts image to grayscale
       calculates a threshold value given the image
       clears all objects touching the border of the image
       labels each object ignoring any touching the border
       finds the smallest raisin and prints out its area and centroid
       prints out the number of raisins not touching border
       calls graph function'''
    #reads in image
    image_file = 'raisins.jpg'
    image = io.imread(image_file)
    
    #converts to grayscale
    grayscale_image = rgb2gray(image)
    
    #calculates threshold value and uses it to identify all objects touching the border
    threshold = threshold_otsu(grayscale_image)
    border_objects = closing(grayscale_image < threshold)
    
    #clears all objects touching the border
    clear_objects= clear_border(border_objects)
    
    #labels each image
    label_object = label(clear_objects)
    
    #returns a rgb coloured label for each object ignoring any object touching border
    overlay = label2rgb(label_object, image=grayscale_image, alpha=0)
    
    
    #iterate over each region and store areas and centroids of all valid regions (i.e. regions with area above 50)
    #increment a count variable to store the total number of raisins
    areas = np.array([])
    count = 0
    centroids = np.array([])
    for region in regionprops(label_object):
        if region.area >= 50:
            centroids = np.append(centroids, region.centroid)
            areas = np.append(areas, region.area)
            count += 1
    
    #create variable for the area and centroid of the smallest raisin
    min_area = np.sort(areas)[0]
    min_centroid = np.sort(centroids)[0]

    #print out total number of raisins
    #print out area and centroid of smallest raisin
    print(f"Number of raisins: {count}")
    print(f"Area of smallest raisin: {min_area}\nCentroid of smallest raisin: {min_centroid}")
     
    #calls graph function to display graph
    graph(overlay, label_object, min_area)


#runs code
if __name__ == "__main__":
    main()