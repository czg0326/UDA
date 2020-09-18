import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F

from torch.autograd import Function

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=64, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0,
                    padding=0
                ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        
        self.kernel = nn.Conv1d(self.input_dim, 
                                self.output_dim,
                                self.context_size, 
                                stride=self.stride, 
                                padding=self.padding, 
                                dilation=self.dilation)

        self.nonlinearity = nn.LeakyReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        #print("xshape:",x.size())
        #_, _, d = x.shape
        #assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        
        x = self.kernel(x.transpose(1,2))
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:           
            x = self.bn(x)
        #return x
        return x.transpose(1,2)

class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x

class XTDNN(nn.Module):

    def __init__(
                    self,
                    features_per_frame=64,
                    final_features=1500,
                    embed_features=512,
                    dropout_p=0.0,
                    batch_norm=True,
                    num_classes=1211
                ):
        super(XTDNN, self).__init__()
        self.features_per_frame = features_per_frame
        self.final_features = final_features
        self.embed_features = embed_features
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        tdnn_kwargs = {'dropout_p':dropout_p, 'batch_norm':self.batch_norm}

        self.frame1 = TDNN(input_dim=self.features_per_frame, output_dim=512, context_size=5, dilation=1, **tdnn_kwargs)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2, **tdnn_kwargs)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3, **tdnn_kwargs)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame5 = TDNN(input_dim=512, output_dim=self.final_features, context_size=1, dilation=1, **tdnn_kwargs)

        self.tdnn_list = nn.Sequential(self.frame1, self.frame2, self.frame3, self.frame4, self.frame5)
        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.final_features*2, self.embed_features)
        self.fc_embed2 = nn.Linear(self.embed_features, self.embed_features)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embed_features)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.embed_features, num_classes)

    def forward(self, x):
        x = self.tdnn_list(x)
        x = self.statspool(x)
        x = self.fc_embed(x)
        x = self.fc_embed2(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        feature = x
        return feature,self.classifier(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        print("classifier")
        print(features.size())
        print(res.size())
        return features, res


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
    
    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
        
    def forward_l2(self, x1, x2):
        assert x1.size() == x2.size()
        x1 = self.l2_norm(x1)
        x2 = self.l2_norm(x2)
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
        
        
class PairwiseSimilarity(Function):
    def __init__(self):
        super(PairwiseSimilarity, self).__init__()
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
        
    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        x1 = self.l2_norm(x1)
        x2 = self.l2_norm(x2)
        cos_dis = torch.mul(x1,x2).sum(dim=1)
        return 1.0 - cos_dis
        
class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,padding=0,bias=False)

                     

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]
        #print("x shape:{}".format(x.shape))
        #print("x size:{}".format(x.size()))
        
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        #print("x_flatten shape:{}".format(x_flatten.shape))
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        #print("vlad shape:{}".format(vlad.shape))
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class Conv_Statistic(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, inplances=512, num_clusters=64):
        super(Conv_Statistic, self).__init__()
        self.conv = nn.Conv2d(inplances, num_clusters, kernel_size=(7, 1), bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        mean_vec = torch.mean(x, dim=-1)
        std_vec = torch.std(x, dim=-1)
        statistic = torch.cat(mean_vec, std_vec, -1)
        reduce_statistic = self.conv(statistic)
        return reduce_statistic
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock50(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock50, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockRes2Net(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth = 16,scale = 4):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth/16.0)))
        self.conv1 = conv1x1(inplanes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        if self.scale != 1:
          out = torch.cat((out,spx[self.nums]),1)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockRes2Net_fc(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth = 7,scale = 4):
        super(BasicBlockRes2Net_fc, self).__init__()
        width = int(math.floor(planes * (baseWidth/16.0)))
        self.conv1 = conv1x1(inplanes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        sp0 = spx[0]
        sp0 = self.convs[0](sp0)
        #print("sp0size:",sp0.size())
        sp1 = sp0 + spx[1]
        sp1 = self.convs[1](sp1)
        #print("sp1size:",sp1.size())
        sp2 = sp0 + sp1 + spx[2]
        sp2 = self.convs[2](sp2)
        #print("sp2size:",sp2.size())
        sp3 = sp0 + sp1 + sp2 + spx[3]
        #print("sp3size:",sp3.size())
        
        out = sp0
        out = torch.cat((out,sp1),1)
        out = torch.cat((out,sp2),1)
        out = torch.cat((out,sp3),1)
        
#        for i in range(self.nums):
#          if i==0:
#            sp = spx[i]
#          else:
#            sp = sp + spx[i]
#          sp = self.convs[i](sp)
#          sp = self.relu(self.bns[i](sp))
#          if i==0:
#            out = sp
#          else:
#            out = torch.cat((out,sp),1)
#        if self.scale != 1:
#          out = torch.cat((out,spx[self.nums]),1)
#        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockRes2Net_random(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth = 7,scale = 4):
        super(BasicBlockRes2Net_random, self).__init__()
        width = int(math.floor(planes * (baseWidth/16.0)))
        self.conv1 = conv1x1(inplanes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          flag = random.randint(0,1)
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          if flag == 1:
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        if self.scale != 1:
          out = torch.cat((out,spx[self.nums]),1)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockRes2Net34(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth = 7,scale = 4):
        super(BasicBlockRes2Net34, self).__init__()
        #width = int(math.floor(planes * (baseWidth/16.0)))
        width = int(math.floor(planes/4))

        self.conv1 = conv3x3(inplanes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        if self.scale != 1:
          out = torch.cat((out,spx[self.nums]),1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockRes2Net34_double(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth = 7,scale = 4):
        super(BasicBlockRes2Net34_double, self).__init__()
        #width = int(math.floor(planes * (baseWidth/16.0)))
        inwidth = int(math.floor(inplanes/4))
        outwidth = int(math.floor(planes/4))

        #self.conv1 = conv3x3(inplanes, width * scale, stride)
        #self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        outconvs = []
        bns = []
        for i in range(self.nums):
          convs.append(conv3x3(inwidth,outwidth))
          outconvs.append(conv3x3(outwidth,outwidth))
          bns.append(nn.BatchNorm2d(outwidth))
        self.convs = nn.ModuleList(convs)
        self.outconvs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.inwidth = inwidth
        self.outwidth = outwidth
        self.scale = scale

    def forward(self, x):
        residual = x

        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        spx = torch.split(x, self.inwidth, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            print('sp:',sp.size())
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          print('sp after convs:',sp.size())
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        if self.scale != 1:
          out = torch.cat((out,spx[self.nums]),1)
          
        outspx = torch.split(out, self.outwidth, 1)
        for i in range(self.nums):
          if i==0:
            outsp = outspx[i]
          else:
            outsp = outsp + outspx[i]
          outsp = self.outconvs[i](sp)
          outsp = self.relu(self.bns[i](sp))
          if i==0:
            out = outsp
          else:
            out = torch.cat((out,outsp),1)
        if self.scale != 1:
          out = torch.cat((out,outspx[self.nums]),1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet, self).__init__()
        
        self.relu = ReLU(inplace=True)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0])

        self.inplanes = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.layer2 = self._make_layer(block, 32, layers[1])
        self.inplanes = 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.inplanes = 128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.layer4 = self._make_layer(block, 128, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d([4,1])
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class myResNet_large(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet_large, self).__init__()
        
        self.relu = ReLU(inplace=True)
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, layers[0])

        self.inplanes = 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.inplanes = 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.layer3 = self._make_layer(block, 128, layers[2])
        self.inplanes = 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn4 = nn.BatchNorm2d(256)
        self.layer4 = self._make_layer(block, 256, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d([1,1])
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class myResNet_ori(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet_ori, self).__init__()
        
        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2,bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d([4,1])
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DeepSpeakerModel(nn.Module):
    def __init__(self,embedding_size,num_classes):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size
        #spp net
        self.output_num = [3,2,1]
        self.model = myResNet_ori(BasicBlock, [1, 1, 1, 1])
        self.model.fc = nn.Linear(512*4, self.embedding_size)
        #self.fc1 = nn.Linear(7168,self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        #self.fc_statistic1 = nn.Linear(512*2, self.embedding_size) #4096)
        #self.fc_statistic2 = nn.Linear(4096, self.embedding_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
        
    def spatial_pyramid_pool(self,previous_conv, previous_conv_size, out_pool_size):
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
            w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(x.size(0),-1)# print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp,x.view(x.size(0),-1)), 1)
        return spp
    
    def statistic_layer(self, previous_conv, num_sample = 20):
        x = previous_conv.resize(int(previous_conv.size(0) / num_sample), num_sample, previous_conv.size(1))
        mean_vec = torch.mean(x, dim=1)
        std_vec = torch.std(x, dim=1)
        statistic = torch.cat((mean_vec.view(mean_vec.size(0),-1), std_vec.view(std_vec.size(0), -1)), 1)
        return statistic


    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)
        #normal net
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        features = self.l2_norm(x)
        
        #spp net
        #spp = self.spatial_pyramid_pool(x,[int(x.size(2)),int(x.size(3))],self.output_num)
        #x = self.fc1(spp)
        #features = self.l2_norm(x)
        
        #statistic net
        #x = self.model.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.statistic_layer(x)
        #x = self.fc_statistic1(x)
        #x = self.fc_statistic2(x)
        #features = self.fc_statistic2(x)
        #features = self.l2_norm(x)
        
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=1
        self.features = features*alpha
        
        #x = x.resize(int(x.size(0) / 10),10 , self.embedding_size)
        #self.features =torch.mean(x,dim=1)
        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return features, res

class DeepSpeakerModel_2(nn.Module):
    def __init__(self,embedding_size,num_classes):
        super(DeepSpeakerModel_2, self).__init__()

        self.embedding_size = embedding_size
        #spp net
        self.output_num = [3,2,1]
        self.model = myResNet_large(BasicBlock, [3, 3, 3, 3])
        self.model.fc = nn.Linear(256, self.embedding_size)
        #self.fc1 = nn.Linear(7168,self.embedding_size)
        #self.avgpool =  nn.AdaptiveAvgPool2d([1,1])
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        #self.fc_statistic1 = nn.Linear(512*2, self.embedding_size) #4096)
        #self.fc_statistic2 = nn.Linear(4096, self.embedding_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
        
    def spatial_pyramid_pool(self,previous_conv, previous_conv_size, out_pool_size):
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
            w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(x.size(0),-1)# print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp,x.view(x.size(0),-1)), 1)
        return spp
    
    def statistic_layer(self, previous_conv, num_sample = 20):
        x = previous_conv.resize(int(previous_conv.size(0) / num_sample), num_sample, previous_conv.size(1))
        mean_vec = torch.mean(x, dim=1)
        std_vec = torch.std(x, dim=1)
        statistic = torch.cat((mean_vec.view(mean_vec.size(0),-1), std_vec.view(std_vec.size(0), -1)), 1)
        return statistic


    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)
        #normal net
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        features = self.l2_norm(x)
        
        #spp net
        #spp = self.spatial_pyramid_pool(x,[int(x.size(2)),int(x.size(3))],self.output_num)
        #x = self.fc1(spp)
        #features = self.l2_norm(x)
        
        #statistic net
        #x = self.model.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.statistic_layer(x)
        #x = self.fc_statistic1(x)
        #x = self.fc_statistic2(x)
        #features = self.fc_statistic2(x)
        #features = self.l2_norm(x)
        
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=12.0
        self.features = features*alpha
        
        #x = x.resize(int(x.size(0) / 10),10 , self.embedding_size)
        #self.features =torch.mean(x,dim=1)
        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return features, res

class ResNet(nn.Module):

    def __init__(self, embedding_size, num_classes=1211,dropout_p=0.0,
                    batch_norm=True):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock50, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock50, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock50, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock50, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(512, self.embedding_size)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embedding_size)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print("downsample not none")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        feature = x
        #feature = self.l2_norm(x)
        #scale = 12.0
        #feature = scale * feature
        return feature,self.classifier(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        print("classifier")
        print(features.size())
        print(res.size())
        return features, res

class ResNet_coral(nn.Module):

    def __init__(self, embedding_size, num_classes=1211,dropout_p=0.0,
                    batch_norm=True):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        super(ResNet_coral, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock50, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock50, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock50, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock50, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(512, self.embedding_size)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embedding_size)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.embedding_size, num_classes)
        self.classifier2 = nn.Linear(self.embedding_size, 2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print("downsample not none")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        feature = x
        return feature,self.classifier(feature),self.classifier2(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        print("classifier")
        print(features.size())
        print(res.size())
        return features, res


class Res2Net(nn.Module):

    def __init__(self, embedding_size, baseWidth = 16,scale = 4, num_classes = 1211,dropout_p=0.0,
                    batch_norm=True):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        super(Res2Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlockRes2Net, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockRes2Net, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockRes2Net, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockRes2Net, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(512, self.embedding_size)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embedding_size)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print("downsample not none")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,baseWidth =self.baseWidth,scale = self.scale))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        feature = x
        #feature = self.l2_norm(x)
        #scale = 12.0
        #feature = scale * feature
        return feature,self.classifier(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        print("classifier")
        print(features.size())
        print(res.size())
        return features, res

class Res2Net34(nn.Module):

    def __init__(self, embedding_size, baseWidth = 7,scale = 4, num_classes = 1211,dropout_p=0.0,
                    batch_norm=True):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        super(Res2Net34, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlockRes2Net34, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockRes2Net34, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockRes2Net34, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockRes2Net34, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(128, self.embedding_size)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embedding_size)
        self.drop_embed = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print("downsample not none")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,baseWidth =self.baseWidth,scale = self.scale))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        #print('x1',x.size())
        x = self.conv1(x)
        #print('x2',x.size())
        x = self.bn1(x)
        #print('x3',x.size())
        x = self.relu(x)
        #print('x4',x.size())
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        feature = x
        #feature = self.l2_norm(x)
        #scale = 12.0
        #feature = scale * feature
        return feature,self.classifier(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        print("classifier")
        print(features.size())
        print(res.size())
        return features, res

class ResNet_big(nn.Module):

    def __init__(self, embedding_size, num_classes):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 32
        self.embedding_size = embedding_size
        super(ResNet_big, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 32, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(256, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feature = x
        #feature = self.l2_norm(x)
        #scale = 12.0
        #feature = scale * feature
        return feature,self.classifier(feature)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res
class ResNetCenter(nn.Module):
    def __init__(self, embedding_size, num_classes):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        super(ResNetCenter, self).__init__()
        self.register_buffer('centers', (
                torch.rand(num_classes, embedding_size).to(torch.device("cuda")) - 0.5) * 2)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(128, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feature = x
        #feature = self.l2_norm(x)
        #scale = 12.0
        #feature = scale * feature
        return feature,self.classifier(feature),self.centers
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res
        
class ResNet_vari_cale(nn.Module):

    def __init__(self, embedding_size, num_classes):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        super(ResNet_vari_cale, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
                                bias=False)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        
        self.fc = nn.Linear(128, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)
        #self.scale = scale_factor


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x, scale_factor=1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feature = self.l2_norm(x)
        feature = scale_factor * feature
        return feature
    
    def forward_classifier(self, x, scale_factor=1):
        features = self.forward(x,scale_factor)
        res = self.classifier(features)
        return features, res
        
class ResNet_vlad(nn.Module):

    def __init__(self):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        super(ResNet_vlad, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
                                bias=False)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((4, 1))
        #self.pooling_vlad = NetVLAD(num_clusters=32, dim=128, alpha=1.0)
        
        #self.fc1 = nn.Linear(4096, 1024)
        #self.fc2 = nn.Linear(1024, self.embedding_size)
        #self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape[:2])
        feature = self.avgpool(x)

        return feature

class ThinResNet_vlad(nn.Module):

    def __init__(self):
        self.layers = [2, 3, 3, 3]
        self.inplanes = 64
        super(ThinResNet_vlad, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 48, self.layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 64, self.layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 128, self.layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, self.layers[3], stride=2)
        #self.maxpool2 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(2,1), stride=1, padding=0,
                                bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape[:2])
        feature = self.conv2(x)

        return feature
        
class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad, embedding_size, num_classes):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.embedding_size = embedding_size
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(512 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, self.embedding_size),
        )
        #self.embedding_layer = nn.Linear(512*8, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)
        
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.net_vlad(x)
        x = self.embedding_layer(x)
        embedded_x = self.l2_norm(x)
        return embedded_x
        
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res
        
        
class Conv_Statistic_reduce(nn.Module):
    def __init__(self, inplances=512, num_clusters=64):
        super(Conv_Statistic_reduce, self).__init__()
        self.conv = BasicConv2d(inplances, num_clusters, kernel_size=(3, 1))
       
    def forward(self, x):
        #print("x shape:{}".format(x.shape))
        x = self.conv(x)
        mean_vec = torch.mean(x, dim=-1,keepdim=True)
        std_vec = torch.std(x, dim=-1,keepdim=True)
        #print("mean shape:{}".format(mean_vec.shape))
        statistic = torch.cat((mean_vec, std_vec), 2)
        return statistic 

class ResNet_statistic(nn.Module):

    def __init__(self, embedding_size, num_classes):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        super(ResNet_statistic, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
                                bias=False)
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, self.layers[3], stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.statistic_layer = Conv_Statistic_reduce(128,64)
        self.fc = nn.Linear(256, self.embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #print(x.shape)
        x = self.statistic_layer(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feature = self.l2_norm(x)
        scale = 12.0
        feature = scale * feature
        return feature
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)#, momentum=0.1, affine=True)
        self.relu=ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
