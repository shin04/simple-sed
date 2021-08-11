import yaml

import torch
import torch.nn as nn
import torchaudio.transforms as audio_nn
from torchlibrosa.augmentation import SpecAugmentation

from torchinfo import summary
from utils.scaler import TorchScaler


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


class BidirectionalGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.
    ) -> None:
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(input_feat)
        return recurrent


class CRNN(nn.Module):
    def __init__(
        self,
        sr: int,
        n_filters: int,
        n_window: int,
        hop_length: int,
        n_mels: int,
        cnn_cfg: dict = {},
        rnn_cfg: dict = {},
        dropout_rate: float = 0.5,
        out_features: int = 10,
        attention: bool = True,
    ) -> None:
        super(CRNN, self).__init__()

        self.mel_spec_trans = audio_nn.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_filters,
            win_length=n_window,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amp_to_db = audio_nn.AmplitudeToDB(stype='amplitude')
        self.amp_to_db.amin = 1e-5

        self.scaler = TorchScaler('instance', 'minmax', dims=[1, 2])

        # self.spec_aug = SpecAugmentation(
        #     time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        # )

        self.cnn = CNN(**cnn_cfg)
        self.rnn = BidirectionalGRU(**rnn_cfg)

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(128 * 2, out_features)
        self.sigmoid = nn.Sigmoid()

        self.attention = attention
        if attention:
            self.att_dense = nn.Linear(128 * 2, out_features)
            self.att_softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        input: waveform (batch_size, frames)
        """

        x = self.mel_spec_trans(input)
        # log_offset = 1e-6
        # x = torch.log(x + log_offset)
        x = self.amp_to_db(x).clamp(min=-50, max=80)
        x = self.scaler(x)
        # x = self.spec_aug(x)

        x = x.transpose(1, 2).unsqueeze(1)

        # (batch_size, channels, freq, frames) > (batch_size, channels, frames, freq)
        # x = x.transpose(3, 2)
        x = self.cnn(x)

        # (batch_size, channels, frames, freq) > (batch_size, frames, channels)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)

        x = self.rnn(x)
        x = self.dropout(x)

        strong_digit = self.dense(x)
        strong = self.sigmoid(strong_digit)

        if self.attention:
            att_x = self.att_dense(x)
            att_x = self.att_softmax(att_x)
            att_x = torch.clamp(att_x, min=1e-7, max=1)
            weak = (strong * att_x).sum(1) / att_x.sum(1)
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak


if __name__ == '__main__':
    with open('../config/urban_sed_2.yaml') as yml:
        conf = yaml.load(yml)

    model_conf = conf['model']
    feat_conf = conf['feature']
    model = CRNN(
        sr=conf['dataset']['sr'],
        n_filters=feat_conf['n_filters'],
        n_window=feat_conf['n_window'],
        hop_length=feat_conf['hop_length'],
        n_mels=feat_conf['n_mels'],
        cnn_cfg=model_conf['cnn'],
        rnn_cfg=model_conf['rnn']
    ).cpu()
    # model = CRNN(
    #     44100, 2048, 2048, 256, 128
    # ).cpu()
    summary(model, input_size=(8, 44100*10))

    # cnn = CNN().cpu()
    # summary(cnn, input_size=(8, 1, 1723, 128))
