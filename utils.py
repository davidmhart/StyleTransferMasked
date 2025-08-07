# You may be able to use a lot of the code in my original utils library,
# but I recommend trying to implement them yourself first to practice.


import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

import torch.nn as nn
import torch.nn.functional as F


def loadImage(filename, asTensor=False):
    '''
    Load an image from a file

    inputs
    filename: path to image file
    asTensor: if True, return image as PyTorch Tensor, else return as NumPy array

    output
    image: image as PyTorch Tensor or NumPy array
    '''
    
    image = Image.open(filename).convert('RGB')
    image = np.array(image)
    image = image.astype(np.float32)/255.0  # Normalize to [0, 1] range
    if asTensor:
        image = imageToTensor(image)
        #image = image.unsqueeze(0)
    return image



def saveImage(image, filename):
    '''
    Save an image to a file

    inputs
    image: image as PyTorch Tensor or NumPy array
    filename: path to save image file
    '''
    
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = tensorToImage(image)
        else:
            raise ValueError("Input image must be a NumPy array or a PyTorch Tensor.")

    # Convert image to [0, 255] range for saving
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(image)
    image.save(filename)



def loadMask(filename, asTensor=False):
    '''
    Load a mask from a file

    inputs
    filename: path to mask file
    asTensor: if True, return mask as PyTorch Tensor, else return as NumPy array

    output
    mask: mask as PyTorch Tensor or NumPy array
    '''
    
    mask = Image.open(filename).convert('L')
    mask = np.array(mask)
    mask = mask.astype(np.float32) / 255.0  # Normalize to [0, 1] range
    if asTensor:
        mask = maskToTensor(mask)
    return mask



def saveMask(mask, filename):
    '''
    Save a mask to a file

    inputs
    mask: mask as PyTorch Tensor or NumPy array
    filename: path to save mask file
    '''
    
    if not isinstance(mask, np.ndarray):
        if isinstance(mask, torch.Tensor):
            mask = tensorToMask(mask)
        else:
            raise ValueError("Input mask must be a NumPy array or a PyTorch Tensor.")

    # Convert mask to [0, 255] range for saving
    mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)

    mask = Image.fromarray(mask)
    mask.save(filename)



def imageToTensor(image,addBatch=True):
    '''
    Convert an image from a NumPy array to a PyTorch Tensor

    inputs
    image: image as NumPy array

    output
    tensor: image as PyTorch Tensor
    '''
    
    image = np.transpose(image, (2, 0, 1))  # Convert HxWxC to CxHxW
    if np.amax(image) > 1.0:
        image = image/255.0
    
    result = torch.from_numpy(image).float()
    if addBatch:
        result = result.unsqueeze(0)
    return result



def tensorToImage(tensor):
    '''
    Convert an image from a PyTorch Tensor to a NumPy array

    inputs
    tensor: image as PyTorch Tensor

    output
    image: image as NumPy array
    '''
    
    if not isinstance(tensor, torch.Tensor):
        #print("Warning: Input is not a PyTorch Tensor. Returning input as is.")
        return tensor

    image = tensor.detach().cpu().numpy()
    image = np.clip(image,0,1)
    image = image.squeeze(0)
    image = np.transpose(image, (1, 2, 0)) # Convert CxHxW to HxWxC
    return image.astype(np.float32)  # Ensure image is float32


def maskToTensor(mask):
    '''
    Convert a mask from a NumPy array to a PyTorch Tensor

    inputs
    mask: mask as NumPy array

    output
    tensor: mask as PyTorch Tensor
    '''
    
    mask = torch.from_numpy(mask).unsqueeze(0).float()

    if torch.amax(mask) > 1.0:
        mask = mask/255.0

    return mask



def tensorToMask(tensor):
    '''
    Convert a mask from a PyTorch Tensor to a NumPy array

    inputs
    tensor: mask as PyTorch Tensor

    output
    mask: mask as NumPy array
    '''
    
    mask = tensor.detach().cpu().squeeze().numpy()
    return mask.astype(np.float32)  # Ensure mask is float32


def plotImage(image, title=None):
    '''
    Plot an image using matplotlib

    inputs
    image: image as NumPy array or PyTorch Tensor
    title: title of the plot
    cmap: colormap to use for the plot (optional)
    '''
    
    if isinstance(image, torch.Tensor):
        image = tensorToImage(image)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()