import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

import json
import numpy as np


class SAMGenerator():

    def __init__(self, model_type="vit_h", checkpoint_path="./models/sam_vit_h_4b8939.pth", device=None):
        '''
        Initialize the SAMGenerator with the specified model type and checkpoint path

        inputs
        model_type: type of the SAM model (default: "vit_h")
        checkpoint_path: path to the SAM model checkpoint (default: "./models/sam_vit_h_4b8939.pth")
        '''
        from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
        from segment_anything.build_sam import sam_model_registry

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.sam.to(device)

    def __call__(self, image):
        '''
        Generate masks using Segment Anything Model

        inputs
        image: image as NumPy array

        output
        jsondata: JSON data as a Python dictionary
        '''
        from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

        # Generate masks using the SAM
        generator = SamAutomaticMaskGenerator(self.sam, output_mode="coco_rle")
        jsondata = generator.generate(image)

        # Ensure the output has an "annotations" key
        if "annotations" not in jsondata:
            jsondata = {"annotations": jsondata}

        return jsondata
    
    



def loadIndices(jsondata, indices, asTensor = True):
    '''
    Load the masks corresponding to the indices from the JSON data

    inputs
    jsondata: JSON data as a Python dictionary
    indices: list of indices to load

    outputs
    masks: list of masks as PyTorch tensors
    '''

    # Check if indices is a single integer and convert it to a list
    if isinstance(indices, int):
        indices = [indices]

    masks = []
    for idx in indices:
        mask = mask_utils.decode(jsondata["annotations"][idx]["segmentation"])
        if asTensor:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        masks.append(mask)
    return masks

def mergeMasks(masks):
    '''
    Combine multiple masks into a single mask

    inputs
    masks: list of masks as PyTorch tensors

    outputs
    combined_mask: combined mask as a PyTorch tensor
    '''
    if len(masks) == 0:
        return None
    
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask += mask

    combined_mask = torch.clamp(combined_mask, 0, 1)  # Ensure values are between 0 and 1

    return combined_mask

def interactiveSelect(image, jsondata):
    '''
    Open an OpenCV window to select masks from the JSON data

    inputs
    image: image as a Numpy array
    jsondata: JSON data as a Python dictionary

    output
    indices: list of indices for corresponding masks
    '''
    indices = []
    clone = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, annotation in enumerate(jsondata["annotations"]):
                mask = mask_utils.decode(annotation["segmentation"])
                #mask = masks[idx]
                if mask[y, x] == 1:
                    if idx not in indices:
                        indices.append(idx)
                    else:
                        indices.remove(idx)
                    display_image = clone.copy()
                    for index in indices:
                        mask = mask_utils.decode(jsondata["annotations"][index]["segmentation"])
                        #mask = masks[index]
                        display_image[mask == 1] = [0, 255, 0]
                    cv2.imshow("image", display_image[:,:,::-1])
                    break
                    # # Display the whole piceces of masks, comment the break
                    
    cv2.imshow("image", image[:,:,::-1])
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return indices


def saveMaskJSON(jsondata, jsonfilename):
    '''
    Save JSON data to a file

    inputs
    jsondata: JSON data as a Python dictionary
    jsonfilename: path to save JSON file
    '''
    with open(jsonfilename, "w") as f:
        json.dump(jsondata, f)


def loadMaskJSON(jsonfilename):
    '''
    Load a presaved JSON file with mask information

    inputs
    jsonfilename: path to JSON file

    output
    jsondata: JSON data as a Python dictionary
    '''
    # f = open(jsonfilename, "r")
    # jsondata = json.load(f)

    with open(jsonfilename, "r") as f:
        jsondata = json.load(f)
    
    return jsondata
