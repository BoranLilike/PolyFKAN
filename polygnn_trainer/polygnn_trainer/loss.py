from torch import nn
import torch
import numpy as np
import random

# fix random seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

class sh_mse_loss_alpha_reg(nn.Module):
    """
    sh_mse_loss + alpha L1 正则化。

    loss = MSE(pred, y) + lambda_alpha * Σ sigmoid(alpha_logit_i)

    用法:
        loss_obj = sh_mse_loss_alpha_reg(lambda_alpha=0.01)
        loss_obj.register_model(model)  # 训练前调用
        # 之后和 sh_mse_loss 一样使用
    """

    def __init__(self, lambda_alpha=0.01):
        super().__init__()
        self.mse_fn = nn.MSELoss()
        self.lambda_alpha = lambda_alpha
        self._model_ref = None

    def register_model(self, model):
        """注册模型引用。每次创建新子模型时需要重新注册。"""
        self._model_ref = model

    def forward(self, predictions, data):
        # 完全相同的 MSE 计算
        predictions = predictions.view(data.num_graphs,)
        data.y = data.y.view(data.num_graphs,)
        mse = self.mse_fn(predictions, data.y)

        # alpha L1 正则
        if self._model_ref is not None and self.lambda_alpha > 0:
            alpha_reg = self._model_ref.get_alpha_reg_loss()
            return mse + self.lambda_alpha * alpha_reg

        return mse
