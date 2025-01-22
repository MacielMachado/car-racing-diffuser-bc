import torch
import torch.nn as nn
import numpy as np


class Model_cnn_mlp(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None, cnn_out_dim=1152):
        super(Model_cnn_mlp, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))
        self.cnn_out_dim = cnn_out_dim

        # Add MLP for dimensionality reduction
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Output size is 3
        )

    def forward(self, x):
        x_embed = self.embed_context(x)  # [batch_size, cnn_out_dim]
        x_out = self.mlp(x_embed)  # Reduce to [batch_size, 3]
        return x_out

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        x_embed = self.imageembed(x3)  # [batch_size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)  # [batch_size, 128]
        return x_embed


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                x = x + x2
            else:
                x = x1 + x2
            return x / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
