import yaml

import torch
import torch.nn as nn
from torchinfo import summary

from model.cnn import CNN
from model.gru import BidirectionalGRU


class HuCRNN(nn.Module):
    def __init__(
        self,
        cnn_cfg: dict = {},
        rnn_cfg: dict = {},
        dropout_rate: float = 0.5,
        out_features: int = 10,
        attention: bool = True,
        n_feats: int = 1
    ) -> None:
        super(HuCRNN, self).__init__()

        self.n_feats = n_feats
        if n_feats != 1:
            self.fc1 = nn.Linear(n_feats, 1)

        self.cnn = CNN(**cnn_cfg)
        self.rnn = BidirectionalGRU(**rnn_cfg)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(rnn_cfg['hidden_size'] * 2, out_features)
        self.sigmoid = nn.Sigmoid()

        self.attention = attention
        if attention:
            self.att_dense = nn.Linear(
                rnn_cfg['hidden_size'] * 2, out_features)
            self.att_softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        input: 
            when use multi layer feature,
                HuBERT feature (batch_size, frame, 768, layer)
            when use single layer feature,
                HuBERT feature (batch_size, frame, 768)
        """

        if self.n_feats == 1:
            x = input.unsqueeze(1)
        else:
            x = self.fc1(input)
            x = x.permute(0, 3, 1, 2)

        x = self.cnn(x)

        # (batch_size, channels, frames, freq) > (batch_size, frames, channels)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)

        x = self.rnn(x)
        x = self.dropout(x)

        strong_digit = self.fc2(x)
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
    with open('../config/hubert.yaml') as yml:
        conf = yaml.load(yml)

    model_conf = conf['model']
    model = HuCRNN(
        cnn_cfg=model_conf['cnn'],
        rnn_cfg=model_conf['rnn'],
        n_feats=12
    ).cpu()
    summary(model, input_size=(8, 499, 768, 12))
