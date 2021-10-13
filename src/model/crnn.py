import yaml

import torch
import torch.nn as nn
from torchinfo import summary

from model.cnn import CNN
from model.gru import BidirectionalGRU


class CRNN(nn.Module):
    def __init__(
        self,
        cnn_cfg: dict = {},
        rnn_cfg: dict = {},
        dropout_rate: float = 0.5,
        out_features: int = 10,
        attention: bool = True,
        layer_init: str = 'default'
    ) -> None:
        super(CRNN, self).__init__()

        self.cnn = CNN(**cnn_cfg)
        self.rnn = BidirectionalGRU(**rnn_cfg)

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(rnn_cfg['hidden_size'] * 2, out_features)
        self.sigmoid = nn.Sigmoid()

        self.attention = attention
        if attention:
            self.att_dense = nn.Linear(
                rnn_cfg['hidden_size'] * 2, out_features)
            self.att_softmax = nn.Softmax(dim=-1)

        self.init_params(layer_init)

    def forward(self, input):
        """
        input: waveform (batch_size, channels, freq, frames)
        """

        # (batch_size, freq, frames) -> (batch_size, channels, frames, freq)
        x = input.transpose(1, 2).unsqueeze(1)

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

    def init_params(self, initialization: str = 'default'):
        """
        reference: ConformerSED

        Parameter:
        ----------
        initialization: str
            default: using pytorch default initialization,
            xavier_uniform: ,
            xavier_normal: ,
            kaiming_uniform: ,
            kaiming_normal: ,
        """

        if initialization.lower() == "default":
            return

        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(
                        f"Unknown initialization: {initialization}")

        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

        # reset some modules with default init
        # for m in self.modules():
        #     if isinstance(m, (torch.nn.Embedding, LayerNorm)):
        #         m.reset_parameters()


if __name__ == '__main__':
    with open('../config/baseline.yaml') as yml:
        conf = yaml.load(yml)

    model_conf = conf['model']
    model = CRNN(
        cnn_cfg=model_conf['cnn'],
        rnn_cfg=model_conf['rnn']
    ).cpu()
    summary(model, input_size=(8, 128, 1000))
    # summary(model, input_size=(8, 768, 499))
