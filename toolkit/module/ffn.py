import torch.nn as nn

class PositionWiseFFN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=None, active_fn=nn.ReLU(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if dim_hidden is None:
            dim_hidden = dim_input
        self.liner1 = nn.Linear(dim_input, dim_hidden)
        self.liner2 = nn.Linear(dim_hidden, dim_output)
        self.actvie_fn = active_fn
    def forward(self, X):
        return self.liner2(self.actvie_fn(self.liner1(X)))