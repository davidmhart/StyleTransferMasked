import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, cdist
import torch
import os
import utils

def _maskValues(image, mask):
    if mask is None:
        return image.reshape(-1, image.shape[-1])
    
    masked_values = image[mask > 0]
    return masked_values.reshape(-1, image.shape[-1])

def _toGray(colors):
    '''
    Convert RGB colors to grayscale using the luminosity method
    '''
    gray = np.dot(colors, [0.2989, 0.5870, 0.1140])
    return gray.reshape(-1, 1)  # Reshape to ensure it has the correct dimensions

def _generateSmallMask(x,mask):
    b, c, h, w = x.shape
    small_mask = mask

    if small_mask.dim() == 2:
        small_mask = small_mask.unsqueeze(0)
    if small_mask.dim() == 3:
        small_mask = small_mask.unsqueeze(0)

    small_mask = torch.nn.functional.interpolate(small_mask, size=(h, w), mode='bilinear', align_corners=False)
    return small_mask.squeeze(0)

def _gramMatrix(features):
    # Assumes flattened input
    b, c, l = features.size()
    G = torch.bmm(features,features.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
    return G.div_(c*l)

def _styleLoss(features,target,mask):
    ib,ic,ih,iw = features.size()
    iF = features.view(ib,ic,-1)
    imask = mask.view(ib,1,-1)
    iF = iF.masked_select(imask.expand_as(iF) > 0)
    iF = iF.view(ib, ic, -1)
    iMean = torch.mean(iF,dim=2)
    iCov = _gramMatrix(iF)

    tb,tc,th,tw = target.size()
    tF = target.view(tb,tc,-1)
    tMean = torch.mean(tF,dim=2)
    tCov = _gramMatrix(tF)

    loss = torch.nn.MSELoss(reduction="sum")(iMean,tMean) + torch.nn.MSELoss(reduction="sum")(iCov,tCov)
    return loss/tb

def getVGGmodel(model_dir="./models", device=None):
    from libs.models_masked import encoder4 as encoder_m
    vgg = encoder_m()
    vgg.load_state_dict(torch.load(os.path.join(model_dir,'vgg_r41.pth')))
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg.to(device)
    vgg.eval()
    return vgg

def computePerceptualStyleLoss(content, style, mask=None, vgg_model=None, style_layers=["r11", "r21", "r31", "r41"], device=None):
    '''
    Compute the perceptual style loss between two images within specified masks
    
    inputs
    content: content image as a NumPy array
    style: style image as a NumPy array
    mask: mask for the image as a NumPy array
    
    output
    loss: Perceptual Style Loss between the two images within the specified masks
    '''
    
    if vgg_model is None:
        vgg_model = getVGGmodel()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(content, np.ndarray):
        content = utils.imageToTensor(content)
    if isinstance(style, np.ndarray):
        style = utils.imageToTensor(style)
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = utils.imageToTensor(mask)

    vgg_model = vgg_model.to(device)
    content = content.to(device)
    style = style.to(device)
    if mask is not None:
        mask = mask.to(device)

    total_style_loss = 0.0  
    tF, _ = vgg_model(content, mask) 
    sF, _ = vgg_model(style)
    
    for layer in style_layers:
        sf_i = sF[layer].detach()
        tf_i = tF[layer].detach()
        small_mask = _generateSmallMask(tf_i, mask)
        total_style_loss += _styleLoss(tf_i, sf_i, small_mask)

    return total_style_loss.item()


def computeEarthMoversDistance(image1, image2, mask1=None, mask2=None, bins=256, in_gray=False):
    '''
    Compute the Earth Mover's Distance (EMD) between two images within specified masks
    
    inputs
    image1: first image as a NumPy array
    image2: second image as a NumPy array
    mask1: mask for the first image as a NumPy array
    mask2: mask for the second image as a NumPy array
    
    output
    emd: Earth Mover's Distance between the two images within the specified masks for each color channel
    '''
    colors1 = _maskValues(image1, mask1)
    #hists1 = [np.histogram(colors1[:, i], bins=bins, range=(0, 1), density=True)[0] for i in range(colors1.shape[1])]
    
    colors2 = _maskValues(image2, mask2)
    #hists2 = [np.histogram(colors2[:, i], bins=bins, range=(0, 1), density=True)[0] for i in range(colors2.shape[1])]
    
    if in_gray:
        colors1 = _toGray(colors1)
        colors2 = _toGray(colors2)

    #emd_values = [wasserstein_distance(hists1[i], hists2[i]) for i in range(len(hists1))]
    emd_values = [wasserstein_distance(colors1[:, i], colors2[:, i]) for i in range(colors1.shape[1])]

    return emd_values


def computeSlicedWassersteinDistance(image1, image2, mask1=None, mask2=None, num_projections=50):
    
    colors1 = _maskValues(image1, mask1)
    colors2 = _maskValues(image2, mask2)

    import ot
    sliced_w_distance = ot.sliced_wasserstein_distance(colors1, colors2, n_projections=num_projections)
    return sliced_w_distance


def computeColorRange(image, mask=None, in_gray=False):
    '''
    Compute the range of color values of an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_range: Color range of the image within the specified mask for each color channel
    '''

    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_range = colors.max(axis=0) - colors.min(axis=0)
    return color_range


def computeColorAverage(image, mask=None, in_gray=False):
    '''
    Compute the average color of an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_ave: Average color of the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_ave = colors.mean(axis=0)
    return color_ave

def computeColorVariance(image, mask=None, in_gray=False):
    '''
    Compute the variance of color distributions in an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_var: Variance of color in the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_var = colors.var(axis=0)
    return color_var

def computeColorSTD(image, mask=None, in_gray=False):
    '''
    Compute the standard deviation of color distributions in an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_std: Standard Deviation of color in the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_std = colors.std(axis=0)
    return color_std

def computeColorSkewness(image, mask=None, in_gray=False):
    '''
    Compute the skewness of color distributions in an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_skew: Skewness of color in the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_mean = colors.mean(axis=0)
    color_std = colors.std(axis=0)
    color_skew = ((colors - color_mean) ** 3).mean(axis=0) / (color_std ** 3)
    return color_skew

def computeColorKurtosis(image, mask=None, in_gray=False):
    '''
    Compute the kurtosis of color distributions in an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image

    output
    color_kurtosis: Kurtosis of color in the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    color_mean = colors.mean(axis=0)
    color_std = colors.std(axis=0)
    color_kurtosis = ((colors - color_mean) ** 4).mean(axis=0) / (color_std ** 4) - 3
    return color_kurtosis


def computeColorHistogram(image, mask=None, bins=256, in_gray=False):
    '''
    Compute the histogram of color distributions in an image within a specified mask
    
    inputs
    image: image as a NumPy array
    mask: mask for the image as a NumPy array, if None is provided, compute on the whole image
    bins: number of bins for the histogram

    output
    hist: Histogram of color in the image within the specified mask for each color channel
    '''
    colors = _maskValues(image, mask)

    if in_gray:
        colors = _toGray(colors)

    hist = [np.histogram(colors[:, i], bins=bins, range=(0, 1))[0] for i in range(colors.shape[1])]
    return np.array(hist)


#def computeEnergyDistance(image1, image2, mask1=None, mask2=None):
#    '''
#    Compute the Earth Mover's Distance (EMD) between two images within specified masks
#    
#    inputs
#    image1: first image as a NumPy array
#    image2: second image as a NumPy array
#    mask1: mask for the first image as a NumPy array
#    mask2: mask for the second image as a NumPy array
#    
#    output
#    emd: Earth Mover's Distance between the two images within the specified masks in 3D color space
#    '''
#    
#    colors1 = _maskValues(image1, mask1)
#    #hists1 = [np.histogram(colors1[:, i], bins=bins, range=(0, 1), density=True)[0] for i in range(colors1.shape[1])]
#    
#    colors2 = _maskValues(image2, mask2)
#    #hists2 = [np.histogram(colors2[:, i], bins=bins, range=(0, 1), density=True)[0] for i in range(colors2.shape[1])]
#
#    # Distances between X and Y samples
#    XY = cdist(colors1, colors2, metric='euclidean')
#    
#    # Distances within X and within Y
#    XX = pdist(colors1, metric='euclidean')
#    YY = pdist(color2, metric='euclidean')
#    
#    # Energy distance formula
#    term1 = 2 * np.mean(XY)
#    term2 = np.mean(XX)
#    term3 = np.mean(YY)
#    
#    return term1 - term2 - term3