# -*- ecoding: utf-8 -*-
# @ModuleName: scatt
# @Author: Kandy
# @Time: 2023-11-23 16:20
import torch
from torch.autograd import Function
import torch.nn.functional as F
from models import pointnet2_cls_msg  # 导入你的 PointNet2 模型
import numpy as np
import importlib

class GradCAM(Function):
    @staticmethod
    def forward(ctx, features, grads, target_layer):
        # 保存反向传播时的梯度
        ctx.save_for_backward(grads)
        # 保存目标层的特征图
        ctx.feature = features
        ctx.target_layer = target_layer

        # 计算相对于目标层输出的梯度
        weights = F.adaptive_avg_pool2d(grads, (1, 1))
        gcam = torch.mul(features, weights).sum(dim=1, keepdim=True)
        return F.relu(gcam)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播计算梯度
        grads = ctx.saved_tensors[0]
        target_layer = ctx.target_layer
        feature = ctx.feature

        weights = F.adaptive_avg_pool2d(grads, (1, 1))
        gcam = torch.mul(feature, weights).sum(dim=1, keepdim=True)

        grad_input = torch.mul(gcam, grads)
        return grad_input, None, None

def pointnet_gradcam(model, input_data, target_class, target_layer):
    # 设置模型为评估模式
    model.eval()

    # 前向传播获取目标层的特征图
    features = model.get_interested_layer_features(input_data, target_layer)

    # 计算损失，反向传播
    model.zero_grad()
    loss = model.compute_loss(input_data, target_class)
    loss.backward()

    # 获取目标层梯度
    grads = model.get_interested_layer_grads(target_layer)

    # 使用 Grad-CAM 可视化
    gcam = GradCAM.apply(features, grads, target_layer)

    # 可视化代码，例如使用 Matplotlib 显示 gcam

# 示例用法
# 加载 PointNet2 模型
model = pointnet2_cls_msg.get_model(8,False)
checkpoint = torch.load(r"C:\Users\Administrator\Desktop\plane\log\classification\pointnet2_cls_msg\checkpoints\best_model.pth")  # 替换为你的模型文件路径
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 读取点云数据
file_path = r"E:\project\PointNet++\data\ship\carrier\carrier_000000.txt"
data = np.loadtxt(file_path, delimiter=' ')  # 假设数据以空格分隔

# 将数据转换为 PyTorch 张量
input_data = torch.tensor(data, dtype=torch.float32)

# 调整数据形状，使其符合 PointNet2 模型的输入要求
input_data = input_data.view(1, 6, -1)  # 假设数据有10000行，每行有6列

# 示例用法
target_class = 0  # 替换为你的目标类别
target_layer = 'conv1'  # 替换为你想要可视化的目标层

# 使用 Grad-CAM 可视化
pointnet_gradcam(model, input_data, target_class, target_layer)
