import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import random
import numpy as np
import polygnn_trainer as pt

from . import utils as utils
from .std_module import StandardMpModule

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class PseudoDC(StandardMpModule):
    def __init__(self, node_size, E, hps, debug=False):
        super().__init__(hps, aggr="add", node_dim=0)
        self.node_size = node_size
        self.E = E
        self.debug = debug

        self.V = pt.layers.Mlp(self.node_size, self.node_size, self.hps, self.debug)
        self.U = pt.layers.Mlp(
            (self.node_size * 2) + self.E.input_dim,
            self.node_size,
            self.hps,
            self.debug,
        )

    def forward(self, x, edge_index, edge_weight, batch):
        if self.debug:
            print("#####")
        return self.propagate(
            edge_index=edge_index, x=x, edge_weight=edge_weight, batch=batch
        )

    def update(self, aggr_out, x):
        if self.debug:
            print("x:", x[:, 0:10])
            print("aggr_out:", aggr_out[:, 0:10])
            pass
        up = F.elu(torch.cat((aggr_out, x), 1))
        out = self.U(up)
        return out

    def message(self, x_i, x_j, edge_weight, edge_index):
        if self.debug:
            print(edge_index)
            print("x_i:", x_i[:, 0:10])
            print("x_j:", x_j[:, 0:10])
            pass

        if self.debug:
            return torch.cat((x_j, edge_weight), 1)

        m_j = self.V(x_j)
        m_ij = self.E(edge_weight)
        return torch.cat((m_j, m_ij), 1)


class MtConcat_PolyMpnn(pt.std_module.StandardModule):
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding,
        debug,
    ):
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        self.debug = debug

        self.readout_dim = 128
        self.R = pt.layers.my_hidden2(
            self.node_size,
            self.readout_dim,
            hps=self.hps,
        )

        self.E = pt.layers.Mlp(
            edge_size,
            edge_size,
            hps=self.hps,
            debug=self.debug,
        )

        self.dc_layers = nn.ModuleList(
            [
                PseudoDC(
                    self.node_size,
                    self.E,
                    hps=self.hps,
                )
                for _ in range(self.hps.capacity.get_value())
            ]
        )
        if self.debug:
            self.dc_layers[0].debug = True

    def forward(self, x, edge_index, edge_weight, batch):
        if self.debug:
            print("#####")
        x_clone = x.clone().detach()

        for i, layer in enumerate(self.dc_layers):
            if i == 0 or i == 1:
                x = layer(x, edge_index, edge_weight, batch)
            else:
                skip_x = last_last_x + x
                x = layer(skip_x, edge_index, edge_weight, batch)
            if i > 0:
                last_last_x = last_x
            last_x = x

        x = self.R(x + x_clone)

        if self.normalize_embedding:
            x = scatter_mean(x, batch, dim=0)
        else:
            x = scatter_sum(x, batch, dim=0)
        return x
    