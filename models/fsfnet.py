import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np




from models.mos_blocks import ResidualEncoder, ResidualDecoder, DecoderBlock
from models.mos_utils  import get_final_image






class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 
    
    
        

class SFT_1(nn.Module):
    def __init__(self, channels):
        super(SFT_1, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x, y):
        return x * self.scale + y * self.shift

class FreAdjust_1(nn.Module):
    def __init__(self, channels):
        super(FreAdjust_1, self).__init__()
        self.sft = SFT_1(channels)
        self.id = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=1)
        )
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels)
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=1)
        )
        self.pre1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels)
        )
        self.pre2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels)
        )

    def forward(self, msf, panf):
        _, _, H, W = msf.shape
        msF = torch.fft.fftn(self.pre1(msf) + 1e-8, dim=(-2, -1))
        panF = torch.fft.fftn(self.pre2(panf) + 1e-8, dim=(-2, -1))

        msF_amp = torch.abs(msF)
        panF_amp = torch.abs(panF)
        msF_pha = torch.atan2(msF.imag, msF.real)
        panF_pha = torch.atan2(panF.imag, panF.real)

        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], dim=1))
        msF_amp = self.sft(msF_amp, panF_amp)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], dim=1))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        out = torch.abs(torch.fft.ifftn(out, s=(H, W), dim=(-2, -1)))
        out = self.post(out)

        msf = self.id(msf)
        out = out + msf

        return out
        
        
   

class FreBlock9(nn.Module):
    #def __init__(self, channels, args):
    def __init__(self, channels):
        super(FreBlock9, self).__init__()

        self.fpre = nn.Conv2d(3, channels, 1, 1, 0)
        #self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, 3, 1, 1, 0)   
       # self.post_1 = nn.Conv2d(3, 64, 1, 1, 0)

    def forward(self, x):
        #print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out






"""
在这篇论文中，我们揭示了扩散 U-Net 的潜力，它被视为一种“免费午餐”，可以在生成过程中大幅提高质量。
我们首先研究了 U-Net 架构对去噪过程的关键贡献，并确定其主要骨干主要贡献于去噪，而其跳跃连接主要将高频特征引入解码器模块，导致网络忽略了骨干语义。
基于这一发现，我们提出了一种简单而有效的方法——称为“FreeU”——它可以提高生成质量，而无需额外的训练或微调。我们的关键见解是，战略性地重新加权源自 U-Net 跳跃连接和骨干特征图的贡献
以利用 U-Net 架构的两个组成部分的优势。在图像和视频生成任务上的有希望的结果表明，我们的 FreeU 可以轻松集成到现有的扩散模型中.
"""

def Fourier_filter(x, threshold, scale):
     x_freq = torch.rfft(x, 2, onesided=False)
     x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])

     B, C, H, W, _ = x_freq.shape
     mask = torch.ones((B, C, H, W), device=x.device)

     crow, ccol = H // 2, W // 2
     mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
     x_freq = x_freq * mask.unsqueeze(-1)

     # IFFT
     x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
     x_filtered = torch.irfft(x_freq, 2, onesided=False, signal_sizes=(H, W))
     return x_filtered


class UNetModel_3(nn.Module):
    def __init__(self, model_channels, num_classes=None):
        super().__init__()
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.input_block = nn.Conv2d(3, model_channels, 3, padding=1)
        self.middle_block = nn.Conv2d(model_channels, model_channels, 3, padding=1)
        self.output_block = nn.Conv2d(model_channels, model_channels, 3, padding=1)
        self.final = nn.Conv2d(model_channels, 3, 3, padding=1)  # Ensure output has 3 channels


def timestep_embedding(timesteps,dim,repeat_only=False):
    return torch.randn((timesteps.shape[0], dim))
 



class UNetModel_1(nn.Module):
    def __init__(self, model_channels, num_classes=None):
        super().__init__()
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.input_block = nn.Conv2d(1, model_channels, 3, padding=1)
        self.middle_block = nn.Conv2d(model_channels, model_channels, 3, padding=1)
        self.output_block = nn.Conv2d(model_channels, model_channels, 3, padding=1)
        self.final = nn.Conv2d(model_channels, 1, 3, padding=1)  # Ensure output has 3 channels
        


def timestep_embedding(timesteps,dim,repeat_only=False):
    return torch.randn((timesteps.shape[0], dim))
 

class Free_UNetModel_1(UNetModel_1):
    def __init__(
            self,
            b1,
            b2,
            s1,
            s2,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2
        # Define the time embedding layer
        self.time_embed = nn.Linear(self.model_channels, self.model_channels)

        #if self.num_classes is not None:
        self.label_emb = nn.Embedding(self.num_classes, self.model_channels)


    def forward(self, x, context=None, **kwargs):
        #assert (y is not None) == (
                    #self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        #timesteps=torch.tensor([1]).cuda()
        #t_emb = torch.randn(timesteps.shape[0],self.model_channels)
        #emb = self.time_embed(t_emb)
        #emb = emb + self.label_emb(timesteps)
        hs = []
        h = x
        h = self.input_block(h)  # First convolution
        hs.append(h)
        h = self.middle_block(h)  # Middle convolution
        for module in [self.output_block, self.final]:  # Output convolutions
            h = module(h)

        return h


# 1.5：b1，用于指定 FreeU 中的第一个模块的参数。它控制截断位置，以调整骨干特征的权重。
# 1.2：b2，用于指定 FreeU 中的第二个模块的参数。它控制截断位置，以调整跳跃连接特征的权重。
# 0.8：s1，用于指定 FreeU 中的第一个模块的参数。它控制截断的比例因子，以调整骨干特征的缩放。
# 0.5：s2，用于指定 FreeU 中的第二个模块的参数。它控制截断的比例因子，以调整跳跃连接特征的缩放。


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x

class Attention(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x

class AttentionLePE(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        x = x.permute(0, 2, 3, 1)
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x



class nchwAttentionLePE(nn.Module):
    """
    Attention with LePE, takes nchw input
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x:torch.Tensor):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        B, C, H, W = x.size()
        q, k, v = self.qkv.forward(x).chunk(3, dim=1) # B, C, H, W

        attn = q.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2) @ \
               k.view(B, self.num_heads, self.head_dim, H*W)
        attn = torch.softmax(attn*self.scale, dim=-1)
        attn = self.attn_drop(attn)

        # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
        output:torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)

        output = self.proj_drop(self.proj(output))

        return output

        
        

        
        
        
        
        
        

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
 ##这是一个辅助函数，用于快速创建二维卷积层，其中包括设置卷积核大小、步长、填充和是否使用偏置项。
##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
  ##BasicConv 类定义了一个基础的卷积模块，包括卷积层、批量归一化和激活函数。
class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
   ##
##########################################################################
## ------ Spatial Attention --------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
 ##这行代码的作用是对输入张量 x 在第二维（维度1，如果考虑0为起始索引的话）计算每个样本的最大值和均值，
# #然后将这两个统计结果在新的第二维上组合成一个新的张量。这种处理常用于特征工程，其中最大值和均值可以提供关于原始数据的有用信息。
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        #scale = F.normalize(scale, p=1, dim=(2, 3))  # 对 scale 进行归一化 新加的
        scaled_x = x * scale
        #normalized_x = F.normalize(scaled_x, p=2, dim=(2, 3)) #新加的
        #return normalized_x
        return scaled_x


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(channel // reduction), #新加的
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                #nn.ReLU(inplace=True),   #新加的
                #nn.BatchNorm2d(channel), #新加的
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
##----------  ----------
class DualAttention(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttention, self).__init__()
        modules_body = [conv(3, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias) , nn.BatchNorm2d(num_features=64) ]
        #modules_body = [conv(64, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 =nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)
    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res





##########################################################################
##---------- Sobel Edge Unit ----------  sobel边缘单元
# class Sobelxy(nn.Module):
#     def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(Sobelxy, self).__init__()
#         sobel_filter = np.array([[1, 0, -1],
#                                  [2, 0, -2],
#                                  [1, 0, -1]])
#         self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
#         self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
#         self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
#         self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
#     def forward(self, x):
#         sobelx = self.convx(x)
#         sobely = self.convy(x)
#         x=torch.abs(sobelx) + torch.abs(sobely)   ##计算每个方向梯度的绝对值，并将它们相加，得到总的边缘强度
#         return x  ##返回最终的结果，即图像的边缘强度图。

##########################################################################
##---------- Dual Attention Edge Unit ----------双注意力边缘单元
# class DualEdgeAttention(nn.Module):
#     def __init__(
#             self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
#         super(DualEdgeAttention, self).__init__()
#         modules_body = [conv(1, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
#         self.body = nn.Sequential(*modules_body)
#
#         ## Spatial Attention
#         self.SA = spatial_attn_layer()
#         ## Channel Attention
#         self.CA = ca_layer(n_feat, reduction, bias=bias)
#         ## Sobel Edge
#         self.Edge = Sobelxy(n_feat)
#         self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         res = self.body(x)
#         sa_branch = self.SA(res)
#         ca_branch = self.CA(res)
#         res_att = torch.cat([sa_branch, ca_branch], dim=1)
#         res_att = self.conv1x1(res_att)
#         edge = self.Edge(res)
#         out  = res_att + edge
#         return out


#填充代码，可以将任意尺寸的图片填充至patch_size的整数倍，便于切块
#返回的是填充后的图片和[高上的补丁数，宽上的补丁数，高上的填充，宽上的填充]
class Padding_tensor(nn.Module):
    def __init__(self, patch_size):
        super(Padding_tensor, self).__init__()
        self.patch_size = patch_size
    def forward(self, x):
        b, c, h, w = x.shape
        h_patches = int(np.ceil(h / self.patch_size))
        w_patches = int(np.ceil(w / self.patch_size))
        h_padding = np.abs(h - h_patches * self.patch_size)
        w_padding = np.abs(w - w_patches * self.patch_size)
        reflection_padding = [0, w_padding, 0, h_padding]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x, [h_patches, w_patches, h_padding, w_padding]

#切块的代码，可以将图像切成(b, c, N, patch_size, patch_size)尺寸
#返回的是小块和填充
class PatchEmbed_tensor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.padding_tensor = Padding_tensor(patch_size)
    def forward(self, x):
        b, c, h, w = x.shape
        x, patches_paddings = self.padding_tensor(x)
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        # -------------------------------------------
        patch_matrix = None
        for i in range(h_patches):
            for j in range(w_patches):
                patch_one = x[:, :, i * self.patch_size: (i + 1) * self.patch_size,
                            j * self.patch_size: (j + 1) * self.patch_size]
                # patch_one = patch_one.flatten(1)
                # patch_one = patch_one.unsqueeze(2)
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size, self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = torch.cat((patch_matrix, patch_one), dim=2)
        # patch_matrix  # (b, c, N, patch_size, patch_size)
        return patch_matrix, patches_paddings
#重建的代码，输入patches_tensor=[b, c, N, patch_size, patch_size]，patches_paddings=[h_patches, w_patches, h_padding, w_padding]
#重建后去除了填充
class Recons_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def forward(self, patches_tensor, patches_paddings):
        B, C, N, Ph, Pw = patches_tensor.shape
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        h_padding = patches_paddings[2]
        w_padding = patches_paddings[3]
        assert N == h_patches * w_patches, \
            f"The number of patches ({N}) doesn't match the Patched_embed operation ({h_patches}*{w_patches})."
        assert Ph == self.patch_size and Pw == self.patch_size, \
            f"The size of patch tensor ({Ph}*{Pw}) doesn't match the patched size ({self.patch_size}*{self.patch_size})."

        patches_tensor = patches_tensor.view(-1, C, N, self.patch_size, self.patch_size)
        # ----------------------------------------
        pic_all = None
        for i in range(h_patches):
            pic_c = None
            for j in range(w_patches):
                if j == 0:
                    pic_c = patches_tensor[:, :, i * w_patches + j, :, :]
                else:
                    pic_c = torch.cat((pic_c, patches_tensor[:, :, i * w_patches + j, :, :]), dim=3)
            if i == 0:
                pic_all = pic_c
            else:
                pic_all = torch.cat((pic_all, pic_c), dim=2)
        b, c, h, w = pic_all.shape
        pic_all = pic_all[:, :, 0:(h-h_padding), 0:(w-w_padding)]
        return pic_all


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
        
        
        
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
        
class AdaptiveFusion(nn.Module):
    def __init__(self, channels=64):
        super(AdaptiveFusion, self).__init__()
        # Learnable weights with initialization
        self.weight_x = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.weight_y = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.constant_(self.weight_x, 0.5)  # Start with equal weights
        nn.init.constant_(self.weight_y, 0.5)
        
    def forward(self, x_res, y_res):
        # Softmax to ensure weights sum to 1
        weights = torch.softmax(torch.cat([self.weight_x, self.weight_y], dim=0), dim=0)
        return x_res * weights[0] + y_res * weights[1]
        
        
##########################################################################
class FSFNet(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3): 
        super(FSFNet, self).__init__()
        # Head
        self.head_1 = DualAttention(n_feat, kernel_size)
        self.head_2 = DualAttention(n_feat, kernel_size)
        self.SE = SE_Block(ch_in=64,reduction=16)
        act=nn.PReLU()
        # Body
        n_resblocks = 2
        m_body = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)
        # Tail
        m_tail = [conv(n_feat, 1, kernel_size)]
        self.tail = nn.Sequential(*m_tail)

        self.act = nn.ReLU(True)
        self.residual_encoder = ResidualEncoder(in_nc=3)
        self.tm_decoder = ResidualDecoder(out_nc=1) 
        self.atmos_decoder = ResidualDecoder(out_nc=3)
        self.fusion = AdaptiveFusion(channels=64)  
        self.FreBlock9 = FreBlock9(channels=64)

    def forward(self, x, y):
        fm_1x, fm_2x, fm_3x, fm_4x = self.residual_encoder(x)     
        tmx = self.tm_decoder( fm_4x, fm_3x, fm_2x, fm_1x)                   
        atmosx = self.atmos_decoder( fm_4x, fm_3x, fm_2x, fm_1x)              
        atmosx = F.avg_pool2d(atmosx, (atmosx.shape[2], atmosx.shape[3]))   
        atmosx= atmosx.view(atmosx.shape[0], -1)                         
        coarsemapx = get_final_image(x.detach(), atmosx.detach(), tmx.detach() , tmin=0.15)  
        x_free = self.FreBlock9(coarsemapx)
        y_free = self.FreBlock9(y)
        x_res = self.head_1(x_free)
        y_res = self.head_2(y_free)
        res = self.fusion(x_res,y_res)
        res = self.body(res)
        out = self.tail(res)
        return out , x_free , y_free ,coarsemapx ,tmx ,atmosx 


def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()




    
    

if __name__ == '__main__':
    
    x = torch.randn(1, 3, 496, 304).cuda()
    y = torch.randn(1, 1, 496, 304).cuda()
    
    model = FSFNet(64).cuda()
    outputs = model(x, y)
    model.eval()
    import time

    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(x, x)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(x, x)
                print(out.shape)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
       # print("Running Time: {:.3f}s\n".format(np.mean(result)))