import torch


class XiaModel(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()

        self.gru1 = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True
        )

        self.gru2 = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True
        )

        self.out = torch.nn.Linear(
            dim,
            out_dim
        )

    def forward(self, x, h01=None, h02=None):
        x, h01 = self.gru1(x, h01)
        x, h02 = self.gru2(x, h02)

        x = self.out(x)

        mask = torch.sigmoid(x)

        return mask, h01, h02