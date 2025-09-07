import torch
import torch.nn as nn
import torch.nn.functional as F

from .util_modules import SelfAttention


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attn = SelfAttention(emded_dim=channels, n_heads=1)

    def forward(self, x):
        # x: [batch_size, channels, h, w]
        batch_size, channels, h, w = x.shape

        residual = x.clone()

        # [batch_size, channels, h, w] --> [batch_size, channels, h, w]
        x = self.groupnorm(x)

        # [batch_size, channels, h, w] --> [batch_size, channels, h*w]
        x = x.view(batch_size, channels, h*w)

        # [batch_size, channels, h*w] --> [batch_size, h*w, channels]
        x = x.transpose(1, 2)

        # [batch_size, h*w, channels] --> [batch_size, h*w, channels]
        x = self.attn(x)

        # [batch_size, channels, h*w] --> [batch_size, channels, h*w]
        x = x.transpose(1, 2)

        # [batch_size, channels, h*w] --> [batch_size, channels, h, w]
        x = x.view((batch_size, channels, h, w))

        x += residual
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x.clone()

        x = self.groupnorm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.relu(x)

        return x + self.residual_layer(residual)


class Encoder(nn.Sequential):
    def __init__(self):
        super(Encoder, self).__init__(
            # [batch_size, 3, h, w] --> [batch_size, 128, h, w]
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # [batch_size, 128, h, w] --> [batch_size, 128, h, w]
            ResidualBlock(128, 128),

            # [batch_size, 128, h, w] --> [batch_size, 128, h/2, w/2]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # [batch_size, 128, h/2, w/2] --> [batch_size, 256, h/2, w/2]
            ResidualBlock(128, 256),

            # [batch_size, 256, h/2, w/2] --> [batch_size, 256, h/2, w/2]
            ResidualBlock(256, 256),

            # [batch_size, 256, h, w] --> [batch_size, 256, h/4, w/4]
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # [batch_size, 256, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            ResidualBlock(256, 512),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/8, w/8]
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            AttentionBlock(512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            nn.GroupNorm(32, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            nn.SiLU(),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 8, h/8, w/8]
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # [batch_size, 8, h/8, w/8] --> [batch_size, 8, h/8, w/8]
            nn.Conv2d(8, 8, kernel_size=3, padding=0)
        )

    def forward(self, x):
        # x: [batch_size, 3, h, w]
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # [batch_size, 8, h/8, w/8] --> two tensors [batch_size, 4, h/8, w/8]
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, min=-30, max=20)

        # Reparametrization trick
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std

        # Scale the latent representation
        x *= 0.18215
        return x


class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__(
            # [batch_size, 4, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            AttentionBlock(512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/8, w/8]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/8, w/8] --> [batch_size, 512, h/4, w/4]
            nn.Upsample(scale_factor=2),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/4, w/4]
            ResidualBlock(512, 512),

            # [batch_size, 512, h/4, w/4] --> [batch_size, 512, h/2, w/2]
            nn.Upsample(scale_factor=2),

            # [batch_size, 512, h/2, w/2] --> [batch_size, 512, h/2, w/2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # [batch_size, 512, h/2, w/2] --> [batch_size, 256, h/2, w/2]
            ResidualBlock(512, 256),

            # [batch_size, 256, h/2, w/2] --> [batch_size, 256, h/2, w/2]
            ResidualBlock(256, 256),

            # [batch_size, 256, h/2, w/2] --> [batch_size, 256, h/2, w/2]
            ResidualBlock(256, 256),

            # [batch_size, 256, h/2, w/2] --> [batch_size, 256, h, w]
            nn.Upsample(scale_factor=2),

            # [batch_size, 256, h, w] --> [batch_size, 256, h, w]
            nn.Conv2d(256, 128, kernel_size=3, padding=1),

            # [batch_size, 256, h, w] --> [batch_size, 128, h, w]
            ResidualBlock(256, 128),

            # [batch_size, 128, h, w] --> [batch_size, 128, h, w]
            ResidualBlock(128, 128),

            # [batch_size, 128, h, w] --> [batch_size, 128, h, w]
            ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # [batch_size, 128, h, w] --> [batch_size, 3, h, w]
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [batch_size, 4, h/8, w/8]
        x /= 0.18215

        for module in self:
            x = module(x)

        # [batch_size, 3, h, w]
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded