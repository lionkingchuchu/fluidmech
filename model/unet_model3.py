from unet_parts import *

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.vencode = nn.ConvTranspose2d(1, 1, kernel_size=30)
        self.up1 = (nn.Sequential(nn.ConvTranspose2d(129, 128, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)))
        self.up2 = (nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)))
        self.up3 = (nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)))
        self.outc = (nn.Sequential(nn.ConvTranspose2d(32, 8, kernel_size=3, stride=1),
                                  nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1)))
        
    def forward(self, x, v):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        v_ = self.vencode(v)
        x4 = torch.cat([x4, v_], axis=1)
        x = self.up1(x4)
        x = torch.cat([x,x3], axis=1)
        x = self.up2(x)
        x = torch.cat([x,x2], axis=1)
        x = self.up3(x)
        x = torch.cat([x,x1], axis=1)
        logits = self.outc(x)
        return logits