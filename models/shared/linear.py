from torch import nn


class ResidualLinear(nn.Linear):
    residual_output: bool = True
