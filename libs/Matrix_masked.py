import torch
import torch.nn as nn
from partialconv2d import PartialConv2d

class CNN(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(CNN,self).__init__()
        if(layer == 'r31'):
            # 256x64x64
            self.convs = nn.Sequential(PartialConv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       PartialConv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       PartialConv2d(64,matrixSize,3,1,1))

        elif(layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(PartialConv2d(512,256,3,1,1),
                                       nn.ReLU(inplace=True),
                                       PartialConv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       PartialConv2d(128,matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x,mask=None):
        #out = self.convs(x)

        out = self.convs[0](x,mask)
        out = self.convs[1](out)
        out = self.convs[2](out,mask)
        out = self.convs[3](out)
        out = self.convs[4](out,mask)

        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)

        if mask is not None:
            mask = mask.view(b,1,-1).expand_as(out)
            out = out.masked_select(mask>0)
            out = out.view(b,c,-1)

            divider = out.shape[2]
        else:
            divider = h*w

        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(divider)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)

class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(layer,matrixSize)
        self.cnet = CNN(layer,matrixSize)
        self.matrixSize = matrixSize

        if(layer == 'r41'):
            self.compress = PartialConv2d(512,matrixSize,1,1,0)
            self.unzip = PartialConv2d(matrixSize,512,1,1,0)
        elif(layer == 'r31'):
            self.compress = PartialConv2d(256,matrixSize,1,1,0)
            self.unzip = PartialConv2d(matrixSize,256,1,1,0)
        self.transmatrix = None

    def forward(self,cF,sF,small_mask,trans=True):
        cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)

        if small_mask is not None:
            small_mask_view = small_mask.view(cb,1,-1)
            b,c,_ = cFF.size()
            cFF = cFF.masked_select(small_mask_view.expand_as(cFF) > 0)
            cFF = cFF.view(b, c, -1)
            
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)


        if small_mask is not None:
            mask_locs = torch.where(small_mask.expand_as(cF)>0)
            cF[mask_locs] -= cMean[mask_locs]
        else:
            cF = cF - cMean

        #cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS


        compress_content = self.compress(cF,small_mask)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if small_mask is not None:

            indices = torch.arange(h*w).to(small_mask.device).masked_select(small_mask_view.squeeze() > 0)

            mask_locs_view = small_mask_view.expand_as(compress_content) > 0
            compress_content = compress_content.masked_select(mask_locs_view)
            compress_content = compress_content.view(b, c, -1)


        if(trans):
            cMatrix = self.cnet(cF,small_mask)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            if small_mask is not None:
                values = torch.bmm(transmatrix,compress_content)

                transfeature = torch.zeros(b,c,h*w).to(values.device)
                transfeature[:, :, indices] = values
                transfeature = transfeature.view(b,c,h,w)
                
                out = self.unzip(transfeature.view(b,c,h,w),small_mask)

                out[mask_locs] += sMeanC[mask_locs]

            else:
                transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
                out = self.unzip(transfeature.view(b,c,h,w),small_mask)
                out = out + sMeanC

            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out
