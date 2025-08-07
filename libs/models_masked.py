import torch
import torch.nn as nn
from partialconv2d import PartialConv2d


class encoder4(nn.Module):
    def __init__(self):
        super(encoder4,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = PartialConv2d(3,3,1,1,0,return_mask=True)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = PartialConv2d(3,64,3,1,0,return_mask=True)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = PartialConv2d(64,64,3,1,0,return_mask=True)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = PartialConv2d(64,128,3,1,0,return_mask=True)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = PartialConv2d(128,128,3,1,0,return_mask=True)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = PartialConv2d(128,256,3,1,0,return_mask=True)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = PartialConv2d(256,512,3,1,0,return_mask=True)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28
    def forward(self,x,mask=None,sF=None,matrix11=None,matrix21=None,matrix31=None):
        output = {}
        out,mask = self.conv1(x,mask)
        out = self.reflecPad1(out); mask = self.reflecPad1(mask)
        out,mask = self.conv2(out,mask)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11']); mask = self.reflecPad7(mask)

        out,mask = self.conv3(out,mask)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12']); mask = self.maxPool(mask)
        out = self.reflecPad4(output['p1']); mask = self.reflecPad4(mask)
        out,mask = self.conv4(out,mask)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21']); mask = self.reflecPad7(mask)

        out,mask = self.conv5(out,mask)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22']); mask = self.maxPool2(mask)
        out = self.reflecPad6(output['p2']); mask = self.reflecPad6(mask)
        out,mask = self.conv6(out,mask)
        output['r31'] = self.relu6(out)
        if(matrix31 is not None):
            feature3,transmatrix3 = matrix31(output['r31'],sF['r31'])
            out = self.reflecPad7(feature3); mask = self.reflecPad7(mask)
        else:
            out = self.reflecPad7(output['r31']); mask = self.reflecPad7(mask)
        out,mask = self.conv7(out,mask)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32']); mask = self.reflecPad8(mask)
        out,mask = self.conv8(out,mask)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33']); mask = self.reflecPad9(mask)
        out,mask = self.conv9(out,mask)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34']); mask = self.maxPool3(mask)
        out = self.reflecPad10(output['p3']); mask = self.reflecPad10(mask)
        out,mask = self.conv10(out,mask)
        output['r41'] = self.relu10(out)

        return output,mask


class decoder4(nn.Module):
    def __init__(self):
        super(decoder4,self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = PartialConv2d(512,256,3,1,0,return_mask=True)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56
        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2) # For the masks

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = PartialConv2d(256,256,3,1,0,return_mask=True)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = PartialConv2d(256,128,3,1,0,return_mask=True)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = PartialConv2d(128,128,3,1,0,return_mask=True)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = PartialConv2d(128,64,3,1,0,return_mask=True)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = PartialConv2d(64,64,3,1,0,return_mask=True)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = PartialConv2d(64,3,3,1,0,return_mask=True)

    def forward(self,x,mask=None):

        if mask is not None:
            # To avoid aliasing, the intermediate masks are based on the original mask
            _, _, rows, cols = x.shape
            mask = mask.unsqueeze(0)
            mask1 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            rows, cols = rows*2, cols*2
            mask2 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            rows, cols = rows*2, cols*2
            mask3 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            rows, cols = rows*2, cols*2
            mask4 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
        else:
            mask1 = None
            mask2 = None
            mask3 = None
            mask4 = None

        # decoder

        out = self.reflecPad11(x); mask1 = self.reflecPad11(mask1) if mask1 is not None else None

        out,mask1 = self.conv11(out,mask1)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out); mask2 = self.reflecPad12(mask2) if mask2 is not None else None
        out,mask2 = self.conv12(out,mask2)

        out = self.relu12(out)
        out = self.reflecPad13(out); mask2 = self.reflecPad13(mask2) if mask2 is not None else None
        out,mask2 = self.conv13(out,mask2)
        out = self.relu13(out)
        out = self.reflecPad14(out); mask2 = self.reflecPad14(mask2) if mask2 is not None else None
        out,mask2 = self.conv14(out,mask2)
        out = self.relu14(out)
        out = self.reflecPad15(out); mask2 = self.reflecPad15(mask2) if mask2 is not None else None
        out,mask2 = self.conv15(out,mask2)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out); mask3 = self.reflecPad16(mask3) if mask3 is not None else None
        out,mask3 = self.conv16(out,mask3)
        out = self.relu16(out)
        out = self.reflecPad17(out); mask3 = self.reflecPad17(mask3) if mask3 is not None else None
        out,mask3 = self.conv17(out,mask3)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out); mask4 = self.reflecPad18(mask4) if mask4 is not None else None
        out,mask4 = self.conv18(out,mask4)
        out = self.relu18(out)
        out = self.reflecPad19(out);mask4 = self.reflecPad19(mask4) if mask4 is not None else None
        out,mask4 = self.conv19(out,mask4)
        return out
