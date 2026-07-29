from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import polygnn_trainer as pt
import torch

import polyfkan.layers as layers


class PolyFKAN(pt.std_module.StandardModule):
    """
    混合策略 PolyFKAN:
    - MPNN 消息传递: 标准 MLP
    - Final MLP: 根据 kafn_config 决定模式:
        - 无 kafn_config / num_harmonics=0 → 标准 MLP
        - mode="replace"  → 纯 FourierKAN
        - mode="residual" → 残差 KAN (sigmoid 门控)
    """

    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        graph_feats_dim=0,
        debug=False,
    ):
        super().__init__(hps)
        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        assert isinstance(graph_feats_dim, int)
        self.graph_feats_dim = graph_feats_dim
        self.debug = debug

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        # Final MLP
        final_input_dim = self.mpnn.readout_dim + self.selector_dim + self.graph_feats_dim
        final_output_dim = 32

        kafn_config = getattr(hps, 'kafn_config', None)
        self._kafn_config = kafn_config  
        if kafn_config is not None:
            self.hps.kafn_config = dict(kafn_config)
        if kafn_config and kafn_config.get('num_harmonics', 0) > 0:
            mode = kafn_config.get('mode', 'replace')

            if mode == 'residual':
                self.final_mlp = pt.layers.ResidualFourierKANMlp(
                    input_dim=final_input_dim,
                    output_dim=final_output_dim,
                    hps=self.hps,
                    debug=False,
                )
                self._model_type = "Residual-FourierKAN"
            else:
                self.final_mlp = pt.layers.FourierKANMlp(
                    input_dim=final_input_dim,
                    output_dim=final_output_dim,
                    hps=self.hps,
                    debug=False,
                )
                self._model_type = "Hybrid-FourierKAN"
        else:
            self.final_mlp = pt.layers.Mlp(
                input_dim=final_input_dim,
                output_dim=final_output_dim,
                hps=self.hps,
                debug=False,
            )
            self._model_type = "Standard"

        self.out_layer = pt.layers.my_output(size_in=final_output_dim, size_out=1)

        if kafn_config and kafn_config.get('num_harmonics', 0) > 0:
            try:
                first = self.final_mlp.layers[0]
                if hasattr(first, 'kan') and hasattr(first.kan, 'a_coeffs'):
                    actual_k = first.kan.a_coeffs.shape[-1]
                    expected_k = kafn_config['num_harmonics']
                    if actual_k != expected_k:
                        raise RuntimeError(
                            f"[polyGNN K MISMATCH] expected K={expected_k}, "
                            f"got a_coeffs.shape[-1]={actual_k}. "
                            f"kafn_config NOT propagated to FourierKAN layers."
                        )
            except (AttributeError, IndexError):
                pass

        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        kan_params = sum(p.numel() for p in self.final_mlp.parameters())
        mpnn_params = sum(p.numel() for p in self.mpnn.parameters())
        print(f"[{self._model_type} polyGNN] 总参数: {total_params:,}, "
              f"MPNN参数: {mpnn_params:,}, FinalMLP参数: {kan_params:,}")

    def get_polymer_fps(self, data):
        return self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)

    def forward(self, data):
        data.yhat = self.get_polymer_fps(data)
        data.yhat = F.silu(data.yhat)
        data.yhat = self.assemble_data(data)
        data.yhat = self.final_mlp(data.yhat)
        data.yhat = self.out_layer(data.yhat)
        return data.yhat.view(data.num_graphs, 1)

    # ============================================================
    # Alpha 正则 + 日志方法
    # ============================================================

    def get_alpha_values(self):
        """获取每层 alpha 值, 返回 list[float] 或 None"""
        if hasattr(self.final_mlp, 'get_alpha_values'):
            return self.final_mlp.get_alpha_values()
        return None

    def get_alpha_reg_loss(self):
        """
        获取 alpha L1 正则损失。
        返回 tensor scalar, 对非残差模式返回 0。
        """
        if hasattr(self.final_mlp, 'get_alpha_reg_loss'):
            return self.final_mlp.get_alpha_reg_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def log_alpha_info(self, prefix=""):
        """打印所有层的 alpha 值"""
        alphas = self.get_alpha_values()
        if alphas is None:
            return ""
        parts = [f"L{i}={a:.4f}" for i, a in enumerate(alphas)]
        msg = f"{prefix}[Alpha] {' | '.join(parts)}"
        print(msg)
        return msg
