from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.up1 = (Up(128, 64,bilinear))
        self.up2 = (Up(64, 32 , bilinear))
        self.up3 = (Up(32, 16, bilinear))
        self.outc = (nn.ConvTranspose2d(16, n_classes, kernel_size=4,stride=2,padding=1))

    def forward(self, x, v):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        v = self.vencode(v)
        x = torch.cat([x4, v], axis=1)
        x = self.up1(x,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x)
        return logits
        
    def vencode(self,v,L=16,d=75):
        p = torch.repeat_interleave(torch.arange(2*L,device='cuda'),2)
        p = torch.pow(torch.tensor(2),p) * torch.pi
        v = (v*p.reshape(-1,1,1))
        v[:,::2] = torch.cos(v[:,::2])
        v[:,1::2] = torch.sin(v[:,::2])
        v = v.repeat((1,1,d,d))
        return v
