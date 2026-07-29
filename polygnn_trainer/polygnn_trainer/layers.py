from typing import Optional
from torch import nn
from .std_module import StandardModule
from .utils import get_unit_sequence
import torch
import torch.nn.functional as F
import math
import numpy as np


class my_hidden2(StandardModule):
    """
    Hidden layer with xavier initialization and dropout
    """

    def __init__(self, size_in, size_out, hps):
        super().__init__(hps)
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = hps.activation.get_value()
        self.dropout = nn.Dropout(hps.dropout_pct.get_value())

    def forward(self, x):
        if self.activation is not None:
            return self.dropout(self.activation(self.linear(x)))
        else:
            return self.dropout(self.linear(x))


class my_output(StandardModule):
    """
    Output layer with xavier initialization on weights
    Output layer with target mean (plus noise) on bias.
    """

    def __init__(self, size_in, size_out, target_mean=None):
        super().__init__(None)
        self.size_in, self.size_out = size_in, size_out
        self.target_mean = target_mean

        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.target_mean != None:
            self.linear.bias.data.uniform_(0.99 * target_mean, 1.01 * target_mean)

    def forward(self, x):
        return self.linear(x)


class Mlp(StandardModule):
    """
    A Feed-Forward neural Network that uses DenseHidden layers
    """

    def __init__(self, input_dim, output_dim, hps, debug, unit_sequence=None):
        super().__init__(hps)
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        if unit_sequence:
            self.unit_sequence = unit_sequence
            self.input_dim = self.unit_sequence[0]
            self.output_dim = self.unit_sequence[-1]
            self.hps.capacity.set_value(len(self.unit_sequence) - 2)
        else:
            self.unit_sequence = get_unit_sequence(
                input_dim, output_dim, self.hps.capacity.get_value()
            )
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out_ = self.unit_sequence[ind + 1]
            self.layers.append(
                my_hidden2(
                    size_in=n_units,
                    size_out=size_out_,
                    hps=hps
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================
# Fourier KAN 层
# ============================================================

class FourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_harmonics=4, dropout_pct=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics

        self.a_coeffs = nn.Parameter(torch.empty(output_dim, input_dim, num_harmonics))
        self.b_coeffs = nn.Parameter(torch.empty(output_dim, input_dim, num_harmonics))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_pct)

        self._initialize_weights()

    def _initialize_weights(self):
        std = 1.0 / math.sqrt(self.input_dim * self.num_harmonics)
        nn.init.normal_(self.a_coeffs, 0, std)
        nn.init.normal_(self.b_coeffs, 0, std)

    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        k_values = torch.arange(1, self.num_harmonics + 1,
                                device=x.device, dtype=x.dtype)
        k_values = k_values.view(1, 1, 1, -1)

        kx = x_expanded * k_values
        cos_terms = self.a_coeffs.unsqueeze(0) * torch.cos(kx)
        sin_terms = self.b_coeffs.unsqueeze(0) * torch.sin(kx)

        out = (cos_terms + sin_terms).sum(dim=(-1, -2))
        out = out + self.bias
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class ResidualFourierKANLayer(nn.Module):
    """
    残差式 Fourier-KAN 层:
        output = Activation(Linear(x)) + sigmoid(alpha_logit) * FourierKAN(x)

    alpha 通过 sigmoid 约束在 [0, 1], 不会变负或爆炸。
    """
    def __init__(self, input_dim, output_dim, num_harmonics=4, dropout_pct=0.0,
                 activation=None, init_alpha=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 路径1: 标准线性变换 + 激活 + Dropout
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_pct)

        # 路径2: FourierKAN 残差路径
        self.kan = FourierKANLayer(input_dim, output_dim, num_harmonics, dropout_pct=0.0)

        # sigmoid 门控 (logit 参数化)
        init_logit = math.log(init_alpha / (1.0 - init_alpha + 1e-8))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, x):
        linear_out = self.linear(x)
        if self.activation is not None:
            linear_out = self.activation(linear_out)
        linear_out = self.dropout(linear_out)

        alpha = torch.sigmoid(self.alpha_logit)
        kan_out = self.kan(x)

        return linear_out + alpha * kan_out

    def get_alpha(self):
        """获取当前 alpha 值 (0~1)"""
        return torch.sigmoid(self.alpha_logit).item()

    def alpha_reg_loss(self):
        """Alpha L1 正则项"""
        return torch.sigmoid(self.alpha_logit)


# ============================================================
# MLP 封装
# ============================================================

class FourierKANMlp(StandardModule):
    """纯 FourierKAN MLP"""
    def __init__(self, input_dim, output_dim, hps, debug, unit_sequence=None):
        super().__init__(hps)
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim

        kafn_config = getattr(hps, 'kafn_config', {})
        self.num_harmonics = kafn_config.get('num_harmonics', 4)

        dropout_pct = hps.dropout_pct.get_value() if hasattr(hps.dropout_pct, 'get_value') else 0.0

        if unit_sequence:
            self.unit_sequence = unit_sequence
            self.input_dim = self.unit_sequence[0]
            self.output_dim = self.unit_sequence[-1]
        else:
            self.unit_sequence = get_unit_sequence(
                input_dim, output_dim, self.hps.capacity.get_value()
            )

        self.layers = nn.ModuleList()
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out = self.unit_sequence[ind + 1]
            self.layers.append(
                FourierKANLayer(
                    input_dim=n_units,
                    output_dim=size_out,
                    num_harmonics=self.num_harmonics,
                    dropout_pct=dropout_pct,
                )
            )

        total = sum(p.numel() for p in self.parameters())
        print(f"  [FourierKANMlp] layers={len(self.layers)}, "
              f"unit_sequence={self.unit_sequence}, "
              f"harmonics={self.num_harmonics}, params={total:,}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualFourierKANMlp(StandardModule):
    """
    残差式 FourierKAN MLP:
    每一层: Activation(Linear(x)) + sigmoid(alpha_logit) * FourierKAN(x)
    """
    def __init__(self, input_dim, output_dim, hps, debug, unit_sequence=None):
        super().__init__(hps)
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim

        kafn_config = getattr(hps, 'kafn_config', {})
        self.num_harmonics = kafn_config.get('num_harmonics', 4)
        self.init_alpha = kafn_config.get('init_alpha', 0.1)

        dropout_pct = hps.dropout_pct.get_value() if hasattr(hps.dropout_pct, 'get_value') else 0.0
        activation = hps.activation.get_value() if hasattr(hps.activation, 'get_value') else None

        if unit_sequence:
            self.unit_sequence = unit_sequence
            self.input_dim = self.unit_sequence[0]
            self.output_dim = self.unit_sequence[-1]
        else:
            self.unit_sequence = get_unit_sequence(
                input_dim, output_dim, self.hps.capacity.get_value()
            )

        self.layers = nn.ModuleList()
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out = self.unit_sequence[ind + 1]
            self.layers.append(
                ResidualFourierKANLayer(
                    input_dim=n_units,
                    output_dim=size_out,
                    num_harmonics=self.num_harmonics,
                    dropout_pct=dropout_pct,
                    activation=activation,
                    init_alpha=self.init_alpha,
                )
            )

        total = sum(p.numel() for p in self.parameters())
        alpha_vals = [f"{layer.get_alpha():.4f}" for layer in self.layers]
        print(f"  [ResidualFourierKANMlp] layers={len(self.layers)}, "
              f"unit_sequence={self.unit_sequence}, "
              f"harmonics={self.num_harmonics}, "
              f"init_alpha={self.init_alpha}, "
              f"alpha_values=[{', '.join(alpha_vals)}], "
              f"params={total:,}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_alpha_values(self):
        """返回每层的 alpha 值列表"""
        return [layer.get_alpha() for layer in self.layers
                if isinstance(layer, ResidualFourierKANLayer)]

    def get_alpha_reg_loss(self):
        """返回所有层的 alpha L1 正则损失总和"""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if isinstance(layer, ResidualFourierKANLayer):
                reg = reg + layer.alpha_reg_loss()
        return reg
