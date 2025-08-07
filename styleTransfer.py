import torch
from libs.Matrix import MulLayer
from libs.models import encoder4, decoder4
from libs.Matrix_masked import MulLayer as MulLayer_m
from libs.models_masked import encoder4 as encoder_m, decoder4 as decoder_m
import numpy as np
import utils
import os

class StyleTransferLinear():

    def __init__(self, model_dir="./models", device=None):
        '''
        Initialize the style transfer model with the given model directory and device.

        inputs
        model_dir: directory where the model files are stored
        device: device to run the model on, if None, will use the default device
        '''
        self.model_dir = model_dir
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        enc_ref = encoder4()
        dec_ref = decoder4()
        matrix_ref = MulLayer('r41')

        enc_ref.load_state_dict(torch.load(os.path.join(model_dir,'vgg_r41.pth')))
        dec_ref.load_state_dict(torch.load(os.path.join(model_dir,'dec_r41.pth')))
        matrix_ref.load_state_dict(torch.load(os.path.join(model_dir,'r41.pth'),map_location=torch.device('cpu')))

        self.encoder = enc_ref.to(self.device)
        self.decoder = dec_ref.to(self.device)
        self.matrix = matrix_ref.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.matrix.eval()

    def __call__(self, content, style, mask=None, premask=False, postmask=True):
        '''
        Apply style transfer to the content image using the style image.

        inputs
        content: content image as PyTorch Tensor
        style: style image as PyTorch Tensor
        mask : mask image as PyTorch Tensor, if None is provided, return the standard styled image
        premask: if True, apply mask before style transfer
        postmask: if True, apply mask after style transfer and alpha blend with original background

        output
        styled: styled image as PyTorch Tensor
        '''

        with torch.no_grad():

            # Convert images to tensors if they are numpy arrays
            if isinstance(content, np.ndarray):
                content = utils.imageToTensor(content)
            if isinstance(style, np.ndarray):
                style = utils.imageToTensor(style)
            if isinstance(mask, np.ndarray):
                mask = utils.maskToTensor(mask)

            original = content.clone()

            if premask and mask is not None:
                # Apply mask to content and style images before style transfer
                content = content * mask

            content = content.to(self.device)
            style = style.to(self.device)
            mask = mask.to(self.device) if mask is not None else None
            original = original.to(self.device)


            #### Stylize ####
            cF_ref = self.encoder(content)
            sF_ref = self.encoder(style)
            feature_ref, _ = self.matrix(cF_ref['r41'], sF_ref['r41'])
            result = self.decoder(feature_ref)


            # Interpolate the result to match the content image size
            if result.shape[2:] != content.shape[2:]:
                result = torch.nn.functional.interpolate(result, size=content.shape[2:], mode='bilinear', align_corners=False)

            if postmask and mask is not None:
                # Apply mask after style transfer and alpha blend with original background
                result = result * mask + original * (1 - mask)

            return result



class StyleTransferPartialConv():

    def __init__(self, model_dir="./models", device=None):
        '''
        Initialize the style transfer model with the given model directory and device.

        inputs
        model_dir: directory where the model files are stored
        device: device to run the model on, if None, will use the default device
        '''
        self.model_dir = model_dir
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device

        enc_ref = encoder_m()
        dec_ref = decoder_m()
        matrix_ref = MulLayer_m('r41')

        enc_ref.load_state_dict(torch.load(os.path.join(model_dir,'vgg_r41.pth')))
        dec_ref.load_state_dict(torch.load(os.path.join(model_dir,'dec_r41.pth')))
        matrix_ref.load_state_dict(torch.load(os.path.join(model_dir,'r41.pth'),map_location=torch.device('cpu')))

        self.encoder = enc_ref.to(self.device)
        self.decoder = dec_ref.to(self.device)
        self.matrix = matrix_ref.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.matrix.eval()

    def __call__(self, content, style, mask=None, premask=False, postmask=True):
        '''
        Apply style transfer to the content image using the style image.

        inputs
        content: content image as PyTorch Tensor
        style: style image as PyTorch Tensor
        mask : mask image as PyTorch Tensor, if None is provided, return the standard styled image
        premask: if True, apply mask before style transfer
        postmask: if True, apply mask after style transfer and alpha blend with original background

        output
        styled: styled image as PyTorch Tensor
        '''

        with torch.no_grad():

            # Convert images to tensors if they are numpy arrays
            if isinstance(content, np.ndarray):
                content = utils.imageToTensor(content)
            if isinstance(style, np.ndarray):
                style = utils.imageToTensor(style)
            if isinstance(mask, np.ndarray):
                mask = utils.maskToTensor(mask)

            original = content.clone()

            if premask and mask is not None:
                # Apply mask to content and style images before style transfer
                content = content * mask

            content = content.to(self.device)
            style = style.to(self.device)
            mask = mask.to(self.device) if mask is not None else None
            original = original.to(self.device)


            #### Stylize ####
            cF_ref,small_mask = self.encoder(content,mask)
            sF_ref,_ = self.encoder(style)
            feature_ref,_ = self.matrix(cF_ref['r41'],sF_ref['r41'],small_mask)
            result = self.decoder(feature_ref,mask)



            # Interpolate the result to match the content image size
            if result.shape[2:] != content.shape[2:]:
                result = torch.nn.functional.interpolate(result, size=content.shape[2:], mode='bilinear', align_corners=False)

            if postmask and mask is not None:
                # Apply mask after style transfer and alpha blend with original background
                result = result * mask + original * (1 - mask)

            return result





def style_with_partialConv(content, style, mask=None, premask=False, postmask=True, prefeathering=0, maskexpand=0, decodefeathering=0):
    '''
    Masked style transfer function using Partial Convolution
    Make sure to use the masked library files

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor
    mask: mask image as PyTorch Tensor, if None is provided, return the standard styled image
    premask: if True, apply mask before style transfer
    postmask: if True, apply mask after style transfer and alpha blend with original background
    prefeathering: if > 0, feather the mask before style transfer
    maskexpand: if > 0, expand the mask in each partial convolution
    decodefeathering: if > 0, feather the mask and blend with the original background at each step of the decoder

    output
    styled: styled image as PyTorch Tensor
    '''
    pass


def multistyle_with_partialConv(content, styles, masks=None, premask=False, postmask=True, prefeathering=0, maskexpand=0, decodefeathering=0):
    '''
    Multi-style transfer function using Partial Convolution
    Make sure to use the masked library files

    inputs
    content: content image as PyTorch Tensor
    styles: list of style images as PyTorch Tensors
    masks: list of mask images as PyTorch Tensors, if None is provided, return the standard styled image
    premask: if True, apply mask before style transfer
    postmask: if True, apply mask after style transfer and alpha blend with original background
    prefeathering: if > 0, feather the mask before style transfer
    maskexpand: if > 0, expand the mask in each partial convolution
    decodefeathering: if > 0, feather the mask and blend with the original background at each step of the decoder

    output
    styled: styled image as PyTorch Tensor
    '''
    pass

def samStyler_reimplmented(content, style, mask=None):

    '''
    SAM Styler paper method, GATYS-based optimization with a mask

    inputs
    content: content image as PyTorch Tensor
    style: style image as PyTorch Tensor
    mask: mask image as PyTorch Tensor, if None is provided, return the standard styled image

    output
    styled: styled image as PyTorch Tensor
    '''
    pass

