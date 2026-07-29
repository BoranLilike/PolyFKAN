import torch
import torch.nn as nn
from .rational_triton import RationalTriton1DGroup
from .kat_1dgroup_torch import Rational_CUDA_A_1DGroup
import json
import os
import math

class KAT_Group(nn.Module):
    def __init__(self, num_groups=8, mode="swish", device="cuda", input_dim=None, 
                 group_strategy="auto", complexity_factor=1.0):
        """
        优化的KAT_Group模块

        Args:
            num_groups (int): 基础分组数
            mode (str): 初始化模式
            device (str): 设备
            input_dim (int): 输入维度
            group_strategy (str): 分组策略 ["auto", "fixed", "adaptive"]
            complexity_factor (float): 复杂度因子，控制分组粒度
        """
        super(KAT_Group, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
        self.order = (5, 4)
        self.device = device
        self.group_strategy = group_strategy
        self.complexity_factor = complexity_factor
        self.input_dim = input_dim
        
        # 优化的分组策略
        self.num_groups = self._optimize_group_strategy(num_groups, input_dim, group_strategy, complexity_factor)
        
        self.lb = None
        self.ub = None

        # 预注册参数
        self.register_buffer('epsilon', torch.tensor(1e-6))
        
        # 初始化权重
        self.initialize(mode=mode, device=device)
        
        # 设置有理函数
        if device == "cuda":
            self.rational = RationalTriton1DGroup.apply
        else:
            self.rational = Rational_CUDA_A_1DGroup

    def _optimize_group_strategy(self, num_groups, input_dim, strategy, complexity_factor):
        """优化的分组策略"""
        if input_dim is None or strategy == "fixed":
            return num_groups
            
        if strategy == "auto":
            # 基于输入维度的智能分组
            if input_dim <= 16:
                groups = max(1, input_dim // 4)
            elif input_dim <= 64:
                ratio = (input_dim - 16) / (64 - 16)
                groups = 4 + int(ratio * 4)
            elif input_dim <= 256:
                log_groups = math.log2(input_dim) - 4
                groups = 8 + int(log_groups * 2)
            else:
                groups = 16

            groups = max(2, min(16, groups))    
            # 应用复杂度因子
            groups = int(groups * complexity_factor)
            
        elif strategy == "adaptive":
            groups = self._progressive_adaptive_grouping(input_dim, num_groups)
        else:
            groups = num_groups
            
        return max(1, min(groups, input_dim))
    def _progressive_adaptive_grouping(self, input_dim, target_groups):
        if input_dim <= 16:
            reasonable_range = (1, 4)
        elif input_dim <= 64:
            reasonable_range = (2, 8)
        elif input_dim <= 256:
            reasonable_range = (4, 16)
        else:
            reasonable_range = (8, 16)

        min_groups, max_groups = reasonable_range
        adjusted_target = max(min_groups, min(target_groups, max_groups))

        return self._find_optimal_groups(input_dim, adjusted_target)        

    def _find_optimal_groups(self, input_dim, target_groups):
        """找到最优的可整除分组数"""
        # 优先选择接近目标的分组数
        candidates = []
        
        # 向上寻找
        for i in range(target_groups, min(target_groups + 10, input_dim + 1)):
            if input_dim % i == 0:
                candidates.append((i, abs(i - target_groups)))
                
        # 向下寻找
        for i in range(target_groups - 1, max(1, target_groups - 10), -1):
            if input_dim % i == 0:
                candidates.append((i, abs(i - target_groups)))
                
        if candidates:
            # 选择最接近目标的分组数
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return max(1, target_groups)  # 保底

    def _calculate_group_parameters(self):
        """计算分组参数统计"""
        if self.input_dim:
            params_per_group = self.input_dim // self.num_groups
            remainder = self.input_dim % self.num_groups
            return {
                'total_groups': self.num_groups,
                'params_per_group': params_per_group,
                'remainder_params': remainder,
                'efficiency': 1.0 - (remainder / self.input_dim) if self.input_dim > 0 else 1.0
            }
        return None

    def initialize(self, mode="swish", device="cuda"):
        """优化的权重初始化"""
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            
            mode_data = data[mode]
            self.lb = mode_data.get("lb", -3.0)
            self.ub = mode_data.get("ub", 3.0)
            
            # 分子权重初始化
            init_w_numerator = mode_data.get("init_w_numerator", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            weight_numerator = torch.tensor(init_w_numerator, device=device).view(1, -1).float()
            
            # 分母权重初始化 - 分组特定的初始化
            init_w_denominator = mode_data.get("init_w_denominator", [1.0, 0.0, 0.0, 0.0])
            base_denominator = torch.tensor(init_w_denominator, device=device).float()
            
            # 为不同分组添加小的随机变化，增强表达能力
            weight_denominator = base_denominator.unsqueeze(0).repeat(self.num_groups, 1)
            if self.num_groups > 1:
                # 添加分组特定的变化
                noise = torch.randn_like(weight_denominator) * 0.01
                weight_denominator = weight_denominator + noise
                
            weight_denominator = torch.clamp(weight_denominator, min=0.1, max=10.0)
            
            # 注册参数
            self.weight_numerator = nn.Parameter(weight_numerator, requires_grad=True)
            self.weight_denominator = nn.Parameter(weight_denominator, requires_grad=True)
            
            # 初始化缓存
            self._update_cached_weights()
            
        except Exception as e:
            print(f"Warning: Initialization failed, using fallback: {e}")
            self._fallback_initialize(device)

    def _fallback_initialize(self, device):
        """备用初始化方案"""
        self.lb, self.ub = -3.0, 3.0
        
        # 简单的Swish近似
        self.weight_numerator = nn.Parameter(
            torch.tensor([0.0, 1.0, 0.5, 0.1, 0.01, 0.001], device=device).view(1, -1).float(),
            requires_grad=True
        )
        
        self.weight_denominator = nn.Parameter(
            torch.ones(self.num_groups, 4, device=device).float(),
            requires_grad=True
        )
        self._update_cached_weights()

    def _update_cached_weights(self):
        """更新缓存的权重（用于性能优化）"""
        if self.training:
            # 训练时不缓存，因为权重在变化
            self._cached_weights = None
        else:
            # 推理时缓存权重
            with torch.no_grad():
                safe_denominator = torch.clamp(self.weight_denominator, min=0.1)
                weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
                self._cached_weights = (weight_numerator, safe_denominator)

    def forward(self, input):
        """优化的前向传播"""
        assert input.dim() in [2, 3], "Input tensor must be 2D or 3D"
        
        # 动态调整边界（如果输入范围变化大）
        if self.training and hasattr(self, 'input_stats'):
            current_max = input.max().item()
            current_min = input.min().item()
            # 适度调整边界以适应数据分布
            adaptive_lb = min(self.lb, current_min * 1.1)
            adaptive_ub = max(self.ub, current_max * 1.1)
            input_clamped = torch.clamp(input, min=adaptive_lb, max=adaptive_ub)
        else:
            input_clamped = torch.clamp(input, min=self.lb, max=self.ub)
        
        # 使用缓存的权重
        if self._cached_weights is not None and not self.training:
            weight_numerator, safe_denominator = self._cached_weights
        else:
            safe_denominator = torch.clamp(self.weight_denominator, min=0.1)
            weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
            if not self.training:
                self._update_cached_weights()
        
        # 应用有理函数
        output = self.rational(input_clamped, weight_numerator, safe_denominator, self.num_groups)
        
        return output

    def extra_repr(self):
        """详细的分组信息"""
        group_info = self._calculate_group_parameters()
        info_str = (f'num_groups={self.num_groups}, order={self.order}, '
                   f'device={self.device}, bounds=({self.lb}, {self.ub}), '
                   f'strategy={self.group_strategy}')
        
        if group_info:
            info_str += f', efficiency={group_info["efficiency"]:.3f}'
            
        return info_str