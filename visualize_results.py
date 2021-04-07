import os
import numpy as np
import json
from PIL import Image, ImageDraw

# set the path to the downloaded data (updated for Alex Guerra): 
data_path = '../data/RedLights2011_Medium'

# Where the bounding boxes are
preds_path = 'hw01_preds/'

# Where we will save the drawn images
save_path  = 'results'
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# Load the predicted json of bounding boxes
with open(preds_path + 'preds.json') as json_file:
    data = json.load(json_file)
    
    # For every key in the data
    for key in data:
    
        # read image key using PIL:
        I = Image.open(os.path.join(data_path, key))
        
        # Make this image drawable
        dI = ImageDraw.Draw(I)
        
        # Get the bounding boxes
        bbs = data[key]
        
        # For each bounding box
        for bb in bbs:
        
            # Draw the bounding box on the image
            dI.rectangle(bb)
        
        # Save the resulting figure.
        I.save(os.path.join(save_path, key))
        
        