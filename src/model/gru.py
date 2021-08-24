import torch
import torch.nn as nn


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
