import os
import numpy as np
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def expand_thresh(im, num_pixels):
    ret = np.zeros(im.shape)
    max_x = im.shape[0]
    max_y = im.shape[1]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # If this pixel is true.
            if im[i,j]:
                # Set the surrounding <num_pixels> pixels to true.
                min_x_ind = max(0, i - num_pixels)
                max_x_ind = min(max_x, i + num_pixels)
                min_y_ind = max(0, j - num_pixels)
                max_y_ind = min(max_y, j + num_pixels)
                ret[min_x_ind:max_x_ind,min_y_ind:max_y_ind] = 1
    return ret
    

def detect_red_light(I, name=""):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    # Color of light
    light_color = [255, 215, 150]
    
    # Light threshold values for rgb
    light_thresh = [15, 70, 70]
    
    # Color of red corona
    red_corona_color = [160, 20, 20]
    
    # Corona threshold values for rgb
    corona_thresh = [40, 40, 40]
    
    # Get our RGB channels
    rI = np.array(I[:,:,0], dtype="int16")
    gI = np.array(I[:,:,1], dtype="int16")
    bI = np.array(I[:,:,2], dtype="int16")
    
    # First filter the image based on light color
    rI_f = np.abs(rI - light_color[0]) <= light_thresh[0]
    gI_f = np.abs(gI - light_color[1]) <= light_thresh[1]
    bI_f = np.abs(bI - light_color[2]) <= light_thresh[2]
    
    # Combine all of these together to get a single threshold
    lI = np.logical_and(rI_f, np.logical_and(gI_f, bI_f))
    
    # expand the pixels of the light filter to produce a new filter
    lI_expanded = expand_thresh(lI, 10)
    
    
    # Then we do the red glow filter 
    rI_f2 = np.abs(rI - light_color[0]) <= corona_thresh[0]
    gI_f2 = np.abs(gI - light_color[1]) <= corona_thresh[1]
    bI_f2 = np.abs(bI - light_color[2]) <= corona_thresh[2]
    
    # Make this filter 
    gI = np.logical_and(rI_f2, np.logical_and(gI_f2, bI_f2))
    
    # Now combine this with our expanded filter
    fI = expand_thresh(np.logical_and(gI, lI_expanded), 5)
    print(name)
    print("Lights:", np.sum(lI), "Red:", np.sum(gI), "Final:", np.sum(fI))
    
    # Save so we can visualize all the intermittent results.
    # plt.imshow(lI)
    # plt.savefig("light_thresholds/" + name)
    # plt.imshow(gI)
    # plt.savefig("glow_thresholds/" + name)
    # plt.imshow(fI)
    # plt.savefig("final_thresholds/" + name)
    # plt.imshow(lI_expanded)
    # plt.savefig("expanded_thresholds/" + name)
    
    
    # Find the contours in this final filtered image
    # Note: I could have wrote this myself but I was lazy, and as this is not 
    #       the 'crux' of the algorithm i thought it would be fine to cut a 
    #       slight corner here.
    contours, hierarchy = cv2.findContours(fI.astype('uint8'), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in contours:
        # Get the bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt)
        bb = [x, y, x + w, y + h]
        
        # Add it to our list of bounding boxes.
        bounding_boxes.append(bb)
    print("Bounding boxes:", len(bounding_boxes))
    print()
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data (updated for Alex Guerra): 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions (updated for Alex Guerra): 
preds_path = 'hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I, name=file_names[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
