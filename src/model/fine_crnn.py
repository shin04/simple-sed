from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torchinfo import summary
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.models.hubert import HubertModel, HubertConfig

from model.cnn import CNN
from model.gru import BidirectionalGRU


class FineCRNN(nn.Module):
    def __init__(
        self,
        pretrain_weight_path: Path,
        use_layer: int,
        freeze_layer: int,
        cnn_cfg: dict = {},
        rnn_cfg: dict = {},
        dropout_rate: float = 0.5,
        out_features: int = 10,
        attention: bool = True,
    ):
        super(FineCRNN, self).__init__()

        # TODO: load HuBERT
        hubert_dict = torch.load(pretrain_weight_path)
        hubert_task_cfg = HubertPretrainingConfig(**hubert_dict['cfg']['task'])
        hubert_model_cfg = HubertConfig(**hubert_dict['cfg']['model'])
        hubert_weights = hubert_dict['model']
        self.use_layer = use_layer
        self.freeze_layer = freeze_layer
        self.hubert_model = HubertModel(hubert_model_cfg, hubert_task_cfg, [[str(i) for i in range(504)]])
        self.hubert_model.load_state_dict(hubert_weights)
        self.set_requires_grad(hubert_weights.keys())

        self.cnn = CNN(**cnn_cfg)
        self.rnn = BidirectionalGRU(**rnn_cfg)

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(rnn_cfg['hidden_size'] * 2, out_features)
        self.sigmoid = nn.Sigmoid()

        self.attention = attention
        if attention:
            self.att_dense = nn.Linear(
                rnn_cfg['hidden_size'] * 2, out_features
            )
            self.att_softmax = nn.Softmax(dim=-1)

    def set_requires_grad(self, weight_keys):
        weight_index = -1
        for i, k in enumerate(weight_keys):
            try:
                layer_num = int(k.split('.')[2])
            except IndexError:
                continue
            except ValueError:
                continue

            if layer_num == self.freeze_layer:
                break
            else:
                weight_index = i

        for i, param in enumerate(self.hubert_model.parameters()):
            param.requires_grad = False

            if i == weight_index:
                break

    def forward(self, input):
        # x = self.hubert_model.extract_features(source=input, output_layer=12)
        res = self.hubert_model.forward(
            source=input,
            mask=False,
            features_only=True,
            output_layer=self.use_layer,
        )
        x = res["features"]

        x = x.squeeze(-1)
        x = x.unsqueeze(1)

        x = self.cnn(x)

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
    with open('../config/hubert.yaml') as yml:
        conf = yaml.load(yml)

    model_conf = conf['model']
    model = FineCRNN(
        pretrain_weight_path='/home/kajiwara21/mrnas02/home/models/hubert/mfcc/pretrain_ite2_23/checkpoint_best.pt',
        use_layer=12,
        freeze_layer=10,
        cnn_cfg=model_conf['cnn'],
        rnn_cfg=model_conf['rnn']
    ).cpu()

    summary(model, input_size=(4, 160000))
