import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        eps: float,
        momentum: float,
        dropput_rate: float,
        pooling: list
    ) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm2d(out_ch, eps=eps, momentum=momentum)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropput_rate)
        self.pool = nn.AvgPool2d(tuple(pooling))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)

        return x


class CNN(nn.Module):
    def __init__(
        self,
        num_layers: int = 7,
        in_channels: int = 1,
        conv_filters: list = [16, 32, 64, 128, 128, 128, 128],
        kernel_sizes: list = [3, 3, 3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1, 1, 1],
        paddings: list = [1, 1, 1, 1, 1, 1, 1],
        poolings: list = [(1, 2), (1, 2), (1, 2), (1, 2),
                          (1, 2), (1, 2), (1, 2)],
        dropout_rate: float = 0.5
    ) -> None:
        super(CNN, self).__init__()

        self.num_layers = num_layers

        self.conv_blocks = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = conv_filters[i-1]

            conv_block = ConvBlock(
                in_ch=in_ch,
                out_ch=conv_filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                eps=0.001,
                momentum=0.99,
                dropput_rate=dropout_rate,
                pooling=poolings[i]
            )
            self.conv_blocks.append(conv_block)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_blocks[i](x)

        return x
