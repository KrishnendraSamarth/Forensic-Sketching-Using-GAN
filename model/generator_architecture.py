
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Encoder block used in the U-Net generator."""

    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """Decoder block with optional dropout."""

    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """U-Net style generator used during training (a.k.a. UNetGenerator)."""

    def __init__(self):
        super().__init__()
        # Encoder / downsampling path
        self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 512)
        self.down6 = ConvBlock(512, 512)
        self.down7 = ConvBlock(512, 512)
        self.down8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True))

        # Decoder / upsampling path with skip connections
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        return self.final(u7)


def get_generator(**kwargs):
    """
    Factory function to create a generator instance.
    You can add initialization parameters here if needed.
    
    Args:
        **kwargs: Any parameters your generator needs during initialization
        
    Returns:
        Generator instance
    """
    return Generator(**kwargs)

