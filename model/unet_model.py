from unet_parts1 import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.vencode = nn.ConvTranspose2d(1, 1, kernel_size=18)
        self.up1_1 = nn.ConvTranspose2d(257, 128, kernel_size=2, stride=2)
        self.up1_2 = DoubleConv(256, 128)
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x, v):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        v_ = self.vencode(v)
        x5 = torch.cat([x5, v_], axis=1)
        x = self.up1_1(x5)
        x = F.pad(x, [1, 0,
                        1, 0])
        
        x = self.up1_2(torch.cat([x,x4],axis=1))
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits