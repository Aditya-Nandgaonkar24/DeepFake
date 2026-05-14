import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        avg_out_sp = torch.mean(x, dim=1, keepdim=True)
        max_out_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out_sp, max_out_sp], dim=1)))
        
        return x * spatial_attn

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, freeze_blocks=True):
        super(SpatialFeatureExtractor, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)
        
        self.features = backbone.features
        
        if freeze_blocks:
            # Leave the last few layers to finetune
            for i in range(len(self.features) - 4):
                for param in self.features[i].parameters():
                    param.requires_grad = False

        # CPU OPTIMIZATION: MobileNet-V3-Large yields 960 channels (instead of 1280)
        # Faster matrix multiplications and lower memory footprint!
        self.attention = CBAM(channels=960)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        return torch.flatten(x, 1) # Yields (batch, 960)
