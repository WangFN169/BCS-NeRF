import torch.nn as nn
import torch
import torch.nn.functional as F
from .trident_conv import MultiScaleTridentConv
#
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature, out_feature,norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        #self._make_layer()

    #def _make_layer(self):
        self.layer_1 = nn.Conv2d(self.in_feature, self.out_feature, 1, stride=stride, padding=0, dilation=dilation)
        self.layer_2 = nn.Conv2d(
            self.out_feature, self.out_feature, 1, stride=stride, padding=0, dilation=dilation
        )
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        if not stride == 1 or in_feature != out_feature:
            self.norm3 = norm_layer(out_feature)
        if stride == 1 and in_feature == out_feature:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=stride), self.norm3)
    def forward(self, x: torch.Tensor):
        out = x
        out = self.actvn(self.layer_1(out))
        out = self.actvn(self.layer_2(out))

        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        #out = F.pixel_shuffle(out, 2)

        return self.actvn(out)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=stride, padding=dilation)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=stride, bias=False, padding=dilation)
        self.bn_atten = norm_layer(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        #atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
#
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, bias=False, dilation=dilation, padding=dilation),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, bias=False, dilation=dilation, padding=dilation),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes, 3, bias=False, dilation=dilation, padding=dilation),
            norm_layer(planes),

        )
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), norm_layer(planes))

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride,
                                       norm_layer=norm_layer,
                                       )  # 1/4 or 1/8    ##

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim,
                                                      kernel_size=3,
                                                      strides=strides,
                                                      paddings=1,
                                                      num_branch=self.num_branch,
                                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        #layer2 = PixelShuffleUpsample(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)    # ##
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, keep_midfeat=False):
        out = []

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        if keep_midfeat:  # for SRF
            out.append(x)
        x = self.layer3(x)  # 1/8 or 1/4
        #if(self.num_branch != 1):
            #x=(self.layer3(x)+self.layer2(x))/2
        #else:
            #x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out.append(x)

        return out
