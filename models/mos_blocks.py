# imports
import torch.nn as nn
import torch


class SEAttention(nn.Module):  # it gives channel attention
    def __init__(self, in_channels, reduced_dim=16):  # input_shape ---> output_shape
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),  
            nn.GELU(),  
            nn.Conv2d(reduced_dim, in_channels, 1), 
            nn.Sigmoid(),  
        )

    def forward(self, x):
        return x * self.se(x)  


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1 ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      stride=stride, padding=(self.kernel_size - 1) // 2, groups=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),  
            nn.GELU(), 
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      stride=1, padding=(self.kernel_size - 1) // 2, groups=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),  
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, stride=stride, groups=1, bias=False),
            nn.InstanceNorm2d(self.out_channels), 
            SEAttention(in_channels=self.out_channels), 
        )
        self.act = nn.GELU() 

    def forward(self, x):
        return self.act(self.backbone(x) + self.shortcut(x)) 


class ResidualEncoder(nn.Module):
    def __init__(self, in_nc):
        super(ResidualEncoder, self).__init__()

        self.block1 = ResidualBlock(in_channels=in_nc, out_channels=16, kernel_size=3, stride=1) 
        self.block2 = ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2)  

        self.block3 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2)  
        self.block4 = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)  

        self.block5 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)  


    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        return x2, x3, x4, x5



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.ResidualBlock = ResidualBlock(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3,
                                           stride=1)
       
    def forward(self, feature, skip_f):
        x = self.upsample(feature) 
        x = torch.cat([x, skip_f], axis=1)  
        x = self.ResidualBlock(x) 
        return x


class ResidualDecoder(nn.Module):
    def __init__(self, out_nc):
        super(ResidualDecoder, self).__init__()

        self.block1 = DecoderBlock(in_channels=256, out_channels=128)
        self.block2 = DecoderBlock(in_channels=128, out_channels=64)
        self.block3 = DecoderBlock(in_channels=64, out_channels=32)
        self.block5 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
                                    nn.Conv2d(16, out_nc, 3, 1, 1), 
                                    nn.Tanh()  
                                    )

    def forward(self, x5, x4, x3, x2):
        x = self.block1(x5, x4)             
        x = self.block2(x, x3)                
        x = self.block3(x, x2)                
        x = (self.block5(x) + 1) / 2         
        return x


if __name__ == '__main__':
    # 创建网络实例
    residual_encoder = ResidualEncoder(in_nc=3) 
    tm_decoder = ResidualDecoder(out_nc=1)      
    atmos_decoder = ResidualDecoder(out_nc=3)   
    x = torch.randn(1, 3, 496, 304)
    fm_1x, fm_2x, fm_3x, fm_4x = residual_encoder(x)
    tmx = tm_decoder(fm_4x, fm_3x, fm_2x, fm_1x)
    atmosx = atmos_decoder(fm_4x, fm_3x, fm_2x, fm_1x)
    atmosx = F.avg_pool2d(atmosx, (atmosx.shape[2], atmosx.shape[3]))
    atmosx = atmosx.view(atmosx.shape[0], -1)
    print("tmx.shape: ", tmx.shape) 
    print("atmosx.shape: ", atmosx.shape) 
    
    